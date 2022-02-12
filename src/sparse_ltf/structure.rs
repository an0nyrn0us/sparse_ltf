use lycaon::base_learner::{BaseLearner, Classifier};
use lycaon::data_type::Sample;

use core::f64::consts::{PI, FRAC_PI_2};

use std::rc::Rc;
use std::cell::RefCell;

use std::hash::Hasher;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;

use super::node::*;
// use super::combination::CombIter;
use super::ltf_classifier::*;



/// The struct SparseLtf holds the data structure
/// that holds all the 2-dimensional ltfs,
/// where the 2 dimension are chosen from n choose 2 ways
pub struct SparseLtf {
    pub root: Rc<RefCell<Node>>,
    pub leaf: Rc<RefCell<Node>>,

    // topologically ordered nodes
    pub nodes:  Vec<Rc<RefCell<Node>>>,

    // Mapping from hash value to `SparseLtfClassifier`.
    pub dictionary: HashMap<u64, Box<SparseLtfClassifier>>,
}


impl SparseLtf {
    pub fn init(sample: &Sample<f64, f64>) -> SparseLtf {
        let m = sample.len();
        let dim = sample.feature_len();
        let leaf = Rc::new(Node::new(m));

        let hash_for_none = m as u64 + 1;
        let root = Rc::new(Node::new(0));
        let mut dictionary = HashMap::new();
        // For each combinations of the features,
        // we construct a DD.
        // let half = dim / 2;
        for i in 0..dim {
            for j in (i+1)..dim {
                let sample_2d = sample.iter()
                    .map(|ex| {
                        let x = ex.data.value_at(i);
                        let y = ex.data.value_at(j);
                        (x, y)
                    })
                    .collect::<Vec<_>>();
                construct_dd(
                    &sample, 
                    sample_2d,
                    i,
                    Some(j),
                    &root,
                    &leaf,
                    &mut dictionary
                );
            }
            let sample_2d = sample.iter()
                .map(|ex| {
                    let x = ex.data.value_at(i);
                    (x, 1.0)
                })
                .collect::<Vec<_>>();
            construct_dd(
                &sample, 
                sample_2d,
                i,
                None,
                &root,
                &leaf,
                &mut dictionary
            );

            // In order to avoid the shotage of memory,
            // We merge the nodes at the half point.
            // if i == half {
            //     let _ = root.borrow_mut().set_hash(hash_for_none);
            //     merge(&root);
            // }
        }
        let _ = root.borrow_mut().rehash(hash_for_none);
        merge(&root);

        // Initiate the hash value
        // let hash_for_none = m as u64 + 1;
        // let _ = root.borrow_mut().set_hash(hash_for_none);

        // sort the nodes
        let nodes = topological_sort(&root, &leaf);


        SparseLtf {
            root,
            leaf,
            nodes,
            dictionary,
        }
    }


    /// Initialize the status of `PathInfo` of each nodes.
    /// This function is called in the `impl BaseLearner`.
    #[inline(always)]
    fn init_path_info(&self) {
        for node in self.nodes.iter() {
            node.borrow_mut().path_info.reset();
        }

        self.leaf.borrow_mut().path_info.reset();

        self.root.borrow_mut().path_info.distance = 0.0;

    }

    pub fn count_ltfs(&self) -> u64 {
        let mut mp: HashMap<u64, u64> = HashMap::new();
        mp.insert(self.root.borrow().hash.unwrap(), 1);

        for node in self.nodes.iter() {
            let h = node.borrow().hash.unwrap();
            let path_count = *mp.get(&h).unwrap();

            if let Some(n) = &node.borrow().neg {
                let nh = n.borrow().hash.unwrap();
                match mp.get_mut(&nh) {
                    Some(v) => {
                        *v += path_count;
                    },
                    None => {
                        mp.insert(nh, path_count);
                    }
                }
            }

            if let Some(p) = &node.borrow().pos {
                let ph = p.borrow().hash.unwrap();
                match mp.get_mut(&ph) {
                    Some(v) => {
                        *v += path_count;
                    },
                    None => {
                        mp.insert(ph, path_count);
                    }
                }
            }
        }


        *mp.get(&self.leaf.borrow().hash.unwrap()).unwrap()
    }
}


#[inline(always)]
fn degs_from_vec(vec: Vec<(f64, f64)>) -> Vec<f64> {
    let m = vec.len();
    let mut temp = vec.into_iter()
        .fold(Vec::with_capacity(2*m), |mut vec, (x, y)| {

            let deg = y.atan2(x);

            vec.push(deg);

            if deg > FRAC_PI_2 {
                vec.push(deg - PI);
            } else {
                vec.push(deg + PI);
            }

            vec
        });
    temp.sort_by(|&a, &b| a.partial_cmp(&b).unwrap());

    let head_val = temp[0];
    temp.push(head_val + 2.0 * PI);

    let mut v = head_val;

    let mut temp = temp.into_iter();
    let mut degrees = Vec::with_capacity(2 * m);
    while let Some(v2) = temp.next() {
        if v != v2 {
            degrees.push(v);
            v = v2;
        }
    }
    degrees.push(v);
    degrees
}


fn construct_dd(sample:   &Sample<f64, f64>,
                sample_2d: Vec<(f64, f64)>,
                i:         usize,
                j:         Option<usize>,
                root:     &Rc<RefCell<Node>>,
                leaf:     &Rc<RefCell<Node>>,
                dict:     &mut HashMap<u64, Box<SparseLtfClassifier>>)
{
    let m = sample.len();
    let degrees = degs_from_vec(sample_2d);

    let mut normal_vectors = Vec::with_capacity(degrees.len());
    let mut biases = Vec::with_capacity(degrees.len());

    let mut deg_iter = degrees.into_iter().peekable();
    while let Some(l) = deg_iter.next() {
        if let Some(&r) = deg_iter.peek() {
            let deg = (l + r + PI) / 2.0;

            let (y, x) = deg.sin_cos();

            let normal_vec;
            let bias;
            match j {
                Some(index) => {
                    normal_vec = vec![(i, x), (index, y)];
                    bias = 0.0;
                },
                None => {
                    normal_vec = vec![(i, x)];
                    bias = y;
                }
            };

            normal_vectors.push(normal_vec);
            biases.push(bias);
        }
    }


    let construct_rest_part = move |preds: Vec<(usize, f64)>, leaf|
        -> Rc<RefCell<Node>>
    {
        let mut node = leaf;
        let mut iter = preds.into_iter().rev();
        while let Some((index, prediction)) = iter.next() {
            let parent = Rc::new(Node::new(index));
            if prediction > 0.0 {
                parent.borrow_mut().pos = Some(node);
            } else {
                parent.borrow_mut().neg = Some(node);
            }
            node = parent;
        }
        node
    };


    // Generate a DAG
    for (vector, bias) in normal_vectors.into_iter().zip(biases) {

        // Generate a prediction vector of `vector`
        let f = SparseLtfClassifier::new(vector, bias);
        let predictions = sample.iter()
            .map(|ex| f.predict(&ex.data))
            .collect::<Vec<_>>();


        let hash = predictions2hash(&predictions);

        // Traverse the current DD as long as possible.
        let mut predictions = predictions.into_iter().enumerate();

        let mut node = Rc::clone(&root);
        let mut last_prediction = 0.0;
        while let Some((_, pred)) = predictions.next() {
            let child;
            if pred > 0.0 {
                if let Some(pos) = &node.borrow().pos {
                    child = Rc::clone(&pos);
                } else {
                    last_prediction = pred;
                    break;
                }
            } else {
                if let Some(neg) = &node.borrow().neg {
                    child = Rc::clone(&neg);
                } else {
                    last_prediction = pred;
                    break;
                }
            }
            node = child;
        }

        // If there exists a path to the leaf node, skip this loop.
        if node.borrow().index == m {
            continue;
        }
        let predictions = predictions.collect::<Vec<_>>();

        let child = construct_rest_part(predictions, Rc::clone(&leaf));


        if last_prediction > 0.0 {
            node.borrow_mut().pos = Some(child);
        } else {
            node.borrow_mut().neg = Some(child);
        }



        // Check whether the collosion occured.
        if dict.contains_key(&hash) {
            panic!("Hash Collosion occured!");
        } else {
            dict.insert(hash, Box::new(f));
        }
    }
}


fn predictions2hash(predictions: &[f64]) -> u64 {
    let mut hasher = DefaultHasher::new();

    for p in predictions.iter() {
        let p = if *p > 0.0 { *p } else { 0.0 };
        hasher.write_u64(p as u64);
    }

    // for (i, p) in predictions.iter().enumerate() {
    //     if *p > 0.0 {
    //         hasher.write_u64(i as u64);
    //     } else {
    //         hasher.write_u64(0);
    //     }
    // }

    hasher.finish()
}


#[inline(always)]
fn merge(root: &Rc<RefCell<Node>>)
{
    use std::collections::VecDeque;

    let mut map: HashMap<u64, _> = HashMap::new();
    let mut deque = VecDeque::new();
    map.insert(root.borrow().hash.unwrap(), Rc::clone(root));
    deque.push_back(Rc::clone(root));


    while let Some(node) = deque.pop_front() {
        let option_neg = node.borrow_mut().neg.take();
        if let Some(neg) = option_neg {
            let hash = neg.borrow().hash.unwrap();

            let new_neg;
            if let Some(same_tree) = map.get(&hash) {
                new_neg = Rc::clone(&same_tree);
            } else {
                map.insert(hash, Rc::clone(&neg));
                deque.push_back(Rc::clone(&neg));
                new_neg = neg;
            }
            node.borrow_mut().neg.replace(new_neg);
        }

        let option_pos = node.borrow_mut().pos.take();
        if let Some(pos) = option_pos {
            let hash = pos.borrow().hash.unwrap();

            let new_pos;
            if let Some(same_tree) = map.get(&hash) {
                new_pos = Rc::clone(&same_tree);
            } else {
                map.insert(hash, Rc::clone(&pos));
                deque.push_back(Rc::clone(&pos));
                new_pos = pos;
            }
            node.borrow_mut().pos.replace(new_pos);
        }
    }
}


impl BaseLearner<f64, f64> for SparseLtf {
    fn best_hypothesis(&self,
                       sample: &Sample<f64, f64>,
                       distribution: &[f64])
        -> Box<dyn Classifier<f64, f64>>
    {
        self.init_path_info();

        // Execute the dynamic programming in order to find a longest path.
        // Note that `self.nodes` is already sorted topologically.
        for (node_index, node) in self.nodes.iter().enumerate() {
            let distance = node.borrow().path_info.distance;

            let i = node.borrow().index;

            let d = distribution[i] * sample[i].label;

            if let Some(n) = &node.borrow_mut().neg {
                let temp_distance = distance - d;
                n.borrow_mut().path_info.assign(
                    temp_distance, Some(node_index), -1.0
                );
            }


            if let Some(p) = &node.borrow_mut().pos {
                let temp_distance = distance + d;
                p.borrow_mut().path_info.assign(
                    temp_distance, Some(node_index), 1.0
                );
            }
        }

        let m = sample.len();

        // Compute the best prediction vector
        let mut label = vec![0.0; m];
        {
            let mut node  = &self.leaf;
            while let Some(index) = &node.borrow().path_info.prev_index {
                let source = &self.nodes[*index];
                let i = source.borrow().index;

                label[i] = node.borrow().path_info.pred_label;

                node = source;
            }
        }

        let hash = predictions2hash(&label);
        let best_hypothesis = self.dictionary.get(&hash)
            .unwrap()
            .clone();

        best_hypothesis
    }
}


// #[test]
// fn test_2d() {
//     use super::*;
//     use lycaon::data_type::{Data, to_sample};
// 
//     let examples = vec![
//         Data::Dense(vec![ 0.0,  1.0]),
//         Data::Dense(vec![ 1.0,  0.0]),
//         Data::Dense(vec![ 0.0, -1.0]),
//         Data::Dense(vec![-1.0,  0.0]),
//     ];
// 
//     let labels = vec![1.0, 1.0, -1.0, -1.0];
// 
//     let sample       = to_sample(examples, labels);
//     let distribution = vec![1.0 / 4.0; 4];
// 
//     let sparse_ltf = SparseLtf::init(&sample);
//     println!("Number of LTFs: {}", sparse_ltf.count_ltfs());
// 
//     let f = sparse_ltf.best_hypothesis(&sample, &distribution);
// 
//     // dbg!(&sparse_ltf.roots[0]);
// 
// 
// 
//     for ex in sample.iter() {
//         assert_eq!(f.predict(&ex.data), ex.label);
//     }
// }


#[test]
fn test_3d() {
    use super::*;
    use lycaon::data_type::{Data, to_sample};

    let examples = vec![
        Data::Dense(vec![-1.0,  1.0,  1.0]),
        Data::Dense(vec![ 1.0, -1.0,  1.0]),
        Data::Dense(vec![-1.0, -1.0, -1.0]),
        Data::Dense(vec![-1.0,  1.0, -1.0]),
    ];

    let labels = vec![1.0, 1.0, -1.0, -1.0];

    let sample       = to_sample(examples, labels);
    let distribution = vec![1.0 / 4.0; 4];

    let sparse_ltf = SparseLtf::init(&sample);
    // println!("Number of LTFs: {}", sparse_ltf.count_ltfs());

    let _ = sparse_ltf.best_hypothesis(&sample, &distribution);


    println!("# of nodes: {count}", count = sparse_ltf.nodes.len());
    // dbg!(&sparse_ltf.root);
    // for root in sparse_ltf.roots.iter() {
    //     dbg!(&root);
    // }

    // for ex in sample.iter() {
    //     assert_eq!(f.predict(&ex.data), ex.label);
    // }
}
