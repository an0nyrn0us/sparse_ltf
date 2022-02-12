// OLD VERSION
// This program will be compared to the new one
use lycaon::base_learner::{BaseLearner, Classifier};
use lycaon::data_type::{Sample, DType};
use lycaon::base_learner::dstump::{DStumpClassifier, PositiveSide};


use crate::average::AverageOracle;
use std::rc::Rc;
use std::ops::Deref;
use std::cell::{Ref, RefCell, Cell};
use std::hash::Hasher;
use std::collections::{HashMap, BTreeMap};
use std::collections::hash_map::DefaultHasher;

#[derive(Clone)]
pub(self) struct Edge {
    pub(self) value: f64,        // The value of j'the feature
    pub(self) label: Vec<usize>, // indices of the sample that has `self.value` at j'th feature
                                 // If the data is sparse, we store the indices that has non-zero values in j'th feature
                                 // since the sparse data has a few non-zero value.
}


/// In this program, we construct an NZDD for decision stump.
/// This NZDD consists of NZDDs for each feature.
/// The j'th element of the edge matrix corresponds to
/// an NZDD that corresponds to the decision stump for the j'th feature.
pub(self) type EdgeMatrix = Vec<Vec<Edge>>;


pub struct ZDDStump {
    edge_matrix: EdgeMatrix,
}


impl ZDDStump {
    pub fn init(sample: &Sample<f64, f64>) -> ZDDStump {
        let dim = sample.feature_len();
        let mut edge_matrix = Vec::with_capacity(dim);


        match sample.dtype {
            DType::Sparse => {
                // Construct a edge vector for each feature
                for j in 0..dim {
                    let mut btmap: BTreeMap<i64, BTreeSet<usize>> = BTreeMap::new();
                    let mut zero_set = BTreeSet::new();
                    let zero = unsafe { std::mem::transmute(0.0_f64) };


                    for (i, ex) in sample.iter().enumerate() {
                        let v = unsafe { std::mem::transmute(ex.data.value_at(j)) };
                        if v == zero {
                            zero_set.insert(i);
                        } else {
                            match btmap.get_mut(&v) {
                                Some(set) => { set.insert(i); },
                                None => {
                                    let mut set = BTreeSet::new();
                                    set.insert(i);
                                    btmap.insert(v, set);
                                }
                            }
                        }
                    }

                    if zero_set.len() != sample.len() {
                        btmap.insert(zero, zero_set);
                    }


                    // Convert the pair of `(f64, Vec<usize>)` to `Vec<Edge>`.
                    let mut edges = btmap.into_iter()
                        .fold(Vec::new(), |mut vec, (v, label)| {
                            let value = unsafe { std::mem::transmute(v) };
                            let label = label.into_iter().collect::<Vec<_>>();
                            vec.push(Edge { value, label });
                            vec
                        });
                    edge_matrix.push(edges.clone());

                    edges.reverse();
                    edge_matrix.push(edges);

                }
            },
            DType::Dense => {
                // Construct a edge vector for each feature
                for j in 0..dim {
                    // Construct a BTreeMap
                    // In order to group the indices by value,
                    // I use `std::mem::transmute` to interpret the `f64` value as `i64`.
                    let btmap: BTreeMap<i64, Vec<usize>> = sample.iter()
                        .enumerate()
                        .fold(BTreeMap::new(), |mut tree, (i, ex)| {
                            let v: i64 = unsafe { std::mem::transmute(ex.data.value_at(j)) };

                            match tree.get_mut(&v) {
                                Some(vec) => { vec.push(i); },
                                None => { tree.insert(v, vec![i]); }
                            }
                            tree
                        });


                    // Convert the pair of `(f64, Vec<usize>)` to `Vec<Edge>`.
                    let mut edges = btmap.into_iter()
                        .map(|(v, label)| {
                            let value: f64 = unsafe { std::mem::transmute(v) };
                            Edge { value, label }
                        })
                        .collect::<Vec<Edge>>();
                    edge_matrix.push(edges.clone());

                    edges.reverse();
                    edge_matrix.push(edges);
                }
            }
        }
        ZDDStump { edge_matrix }
    }
}


impl BaseLearner<f64, f64> for ZDDStump {
    fn best_hypothesis(&self, sample: &Sample<f64, f64>, distribution: &[f64]) -> Box<dyn Classifier<f64, f64>> {

        let init_distance = sample.iter()
            .zip(distribution.iter())
            .fold(0.0, |acc, (ex, &d)| acc - ex.label * d);

        let mut best = Info::init(init_distance, sample.feature_len(), 0);

        match sample.dtype {
            DType::Dense => {
                for (j, edges) in self.edge_matrix.iter().enumerate() {
                    let mut distance = init_distance;

                    if j % 2 == 1 { distance *= -1.0; }
                    // let mut distance = 0.0;

                    for (e, edge) in edges.iter().enumerate() {
                        best.update(distance, j, e);

                        let weight = edge.label.iter()
                            .fold(0.0_f64, |acc, &l| acc + sample[l].label * distribution[l]);

                        distance += 2.0 * weight;
                    }

                    best.update(distance, j, edges.len());
                }
            },
            DType::Sparse => {
                for (j, edges) in self.edge_matrix.iter().enumerate() {
                    let mut distance = init_distance;

                    if j % 2 == 1 { distance *= -1.0; }
                    // let mut distance = 0.0;

                    for (e, edge) in edges.iter().enumerate() {
                        best.update(distance, j, e);


                        // Compute the weight on the `edge`
                        let weight = if edge.value != 0.0 {
                            // If `edge.value` is non-zero, compute like the dense case.
                            edge.label.iter()
                                .fold(0.0_f64, |acc, &l| acc + sample[l].label * distribution[l])
                        } else {
                            // If `edge.value` is zero, we regard the `edge.label` as
                            // the complementary set
                            if edge.label.is_empty() {
                                sample.iter()
                                    .zip(distribution.iter())
                                    .fold(0.0_f64, |acc, (ex, &d)| acc + ex.label * d)
                            } else {
                                let label = edge.label.clone()
                                    .into_iter()
                                    .collect::<HashSet<_>>();

                                (0..sample.len()).into_iter()
                                    .zip(distribution.iter())
                                    .filter_map(|(n, d)|
                                        if !label.contains(&n) { Some(n as f64 * d) } else { None }
                                    ).sum::<f64>()
                            }
                        };

                        // Add to the distance
                        distance += 2.0 * weight;
                    }
                    best.update(distance, j, edges.len());
                }
            }
        } // End of match
        let dstump = best.dstump(&self.edge_matrix);
        // println!("DStump [ left_side: {:?}, feature_index: {}, threshold: {} ]", left_side, feature_index, threshold);

        Box::new(dstump)
    }
}


/// This struct is only used to construct a best hypothesis from `ZDDStump`.
#[derive(Debug)]
pub(self) struct Info {
    pub(self) distance: f64,
    pub(self) index: usize,
    pub(self) eps_transit: usize
}


impl Info {
    pub(self) fn init(distance: f64, index: usize, eps_transit: usize) -> Info {
        Info { distance, index, eps_transit }
    }


    pub(self) fn update(&mut self, d: f64, i: usize, e: usize) {
        if self.distance < d {
            self.distance = d;
            self.index = i;
            self.eps_transit = e;
        }
    }


    fn threshold(&self, edge_matrix: &EdgeMatrix) -> f64 {
        let edges = &edge_matrix[self.index];
        let e = self.eps_transit;
        let no_eps_transit = edges.len();

        let l;
        let r;

        // the values in the edges are sorted in the ascending order.
        if self.index % 2 == 0 {
            l = if e == 0 { edges[e].value - 2.0 } else { edges[e-1].value };
            r = if e == no_eps_transit { edges.last().unwrap().value + 2.0 } else { edges[e].value };

        // the values in the edges are sorted in the descending order.
        } else {
            l = if e == 0 { edges[e].value + 2.0 } else { edges[e-1].value };
            r = if e == no_eps_transit { edges.last().unwrap().value - 2.0 } else { edges[e].value };
        }

        (l + r) / 2.0
    }


    pub(self) fn dstump(self, edge_matrix: &EdgeMatrix) -> DStumpClassifier {
        let left_side = if self.index % 2 == 1 { PositiveSide::RHS } else { PositiveSide::LHS };

        let feature_index = self.index / 2;

        let threshold = self.threshold(&edge_matrix);


        // println!("DStump [ left_side: {:?}, feature_index: {}, threshold: {} ]", left_side, feature_index, threshold);
        DStumpClassifier { threshold, feature_index, left_side }
    }
}


// TODO! Test & implement this code
impl AverageOracle<f64, f64> for ZDDStump {
    fn margin_distribution(&self, eta: f64, distribution: &[f64], sample: &Sample<f64, f64>) -> Vec<f64> {
        let em_len = self.edge_matrix.len();

        // -----------------------------------------------------------------------------------
        // We first compute the weights for the input of the weight-pushing algorithm


        // 
        // `weight_matrix` holds the logarithmic weights on edges for the input of the weight-pushing algorithm
        // Initialize `weight_matrix`
        let mut weight_matrix: Vec<Vec<f64>> = Vec::with_capacity(em_len);
        match sample.dtype {
            DType::Sparse => {
                for edges in self.edge_matrix.iter() {
                    let mut val;
                    let mut weights = Vec::with_capacity(edges.len() + 1);

                    for edge in edges.iter() {
                        if edge.value != 0.0 {
                            val = edge.label
                                .iter()
                                .fold(0.0, |acc, &i| acc + sample[i].label * distribution[i]);
                        } else {
                            // If the val ie equals to 0.0, the edge has the indices
                            // that has non-zero value at j'th feature.
                            // If `edge.labels.is_empty()`, all the element has zero-value
                            // at j'th feature.
                            if edge.label.is_empty() {
                                val = distribution.iter()
                                    .zip(sample.iter())
                                    .fold(0.0, |acc, (&d, ex)| acc + d * ex.label);
                            } else {
                                let label = edge.label.clone()
                                    .into_iter()
                                    .collect::<HashSet<_>>();

                                val = (0..sample.len()).filter(|l| !label.contains(&l))
                                    .fold(0.0, |acc, l| acc + sample[l].label * distribution[l]);
                            }
                        }
                        weights.push(2.0 * eta * val);
                    }
                    weight_matrix.push(weights);
                }
            },
            DType::Dense => {
                for edges in self.edge_matrix.iter() {
                    let mut weights = Vec::with_capacity(em_len);
                    for edge in edges.iter() {
                        let val = edge.label.iter().fold(0.0, |acc, &i| acc + sample[i].label * distribution[i]);
                        weights.push(2.0 * eta * val);
                    }
                    weight_matrix.push(weights);
                }
            }
        }


        // 
        // `vertex_weights` holds the weights, where the weight $s(u)$ for node $u$ is
        // defined by $\sum_{e = (u, v)} w(e) \cdot s(v)$, $w(e)$ is the weight on the edge
        // defined in the above initialization.
        let mut vertex_weights: Vec<Vec<f64>> = Vec::with_capacity(em_len);

        // 
        // `max_subroot` is the maximum of the weights in `weight_matrix[j][0]` over `j`.
        let mut max_subroot = f64::MIN;
        for weights in weight_matrix.iter() {
            let mut vweights = Vec::with_capacity(weights.len());

            let mut child_weight = 0.0;
            for &w in weights.iter().rev() {
                // compute the max of $w(e) \cdot s(v)$ for all $e = (u, v)$
                // since each node has two edges and one of them has no label,
                // the weght is zero.
                let temp = w + child_weight;
                let m = if temp > 0.0 { temp } else { 0.0 };


                vweights.push(
                    m + ((temp - m).exp() + (-m).exp()).ln()
                );

                // update `child_weight`
                child_weight = w;
            }
            vweights.reverse();
            if vweights[0] > max_subroot {
                max_subroot = vweights[0];
            }
            vertex_weights.push(vweights);
        }


        // compute the weight on the root.
        let root_weight = vertex_weights.iter()
            .fold(0.0, |acc, vweights| acc + (vweights[0] - max_subroot).exp())
            .ln()
            + max_subroot;



        // End of the initialization of the input of the weight-pushing algorithm
        // -----------------------------------------------------------------------------------
        // Execute weight pushing


        // 
        // `root2subroot` holds the weight on edge
        // that connects from root to sub-root.
        let root2subroots = vertex_weights.iter()
            .map(|vweights| (vweights[0] - root_weight).exp())
            .collect::<Vec<_>>();


        for (vweights, weights) in vertex_weights.iter().zip(weight_matrix.iter_mut()) {
            let mut it    = weights.iter_mut().peekable();
            let mut vw_it = vweights.iter();
            while let (Some(w), Some(vw)) = (it.next(), vw_it.next()) {
                let child = if let Some(&&mut ch) = it.peek() { ch } else { 0.0 };

                *w = (*w + child - vw).exp();
            }
        }


        // End of the weight-pushing algorithm
        // -----------------------------------------------------------------------------------
        // Compute the margin distribution vector


        let mut mdist = vec![0.0; sample.len()];

        for ((&r2s, weights), edges) in root2subroots.iter().zip(weight_matrix.iter()).zip(self.edge_matrix.iter()) {
            let mut probability = r2s;

            for (edge, &weight) in edges.iter().zip(weights.iter()) {
                probability *= weight;
                for &i in edge.label.iter() {
                    mdist[i] += probability;
                }
            }
        }

        for (d, ex) in mdist.iter_mut().zip(sample.iter()) {
            *d = ex.label * (2.0 * *d - 1.0);
        }

        mdist
    }
}



