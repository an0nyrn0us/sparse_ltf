use lycaon::base_learner::{BaseLearner, Classifier};
use lycaon::data_type::{Sample, Data};

use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

use grb::prelude::*;

use super::node::*;
use super::ltf_classifier::*;


pub struct BddLtf {
    pub root:  Rc<RefCell<Node>>,
    pub leaf:  Rc<RefCell<Node>>,
    pub nodes: Vec<Rc<RefCell<Node>>>,

    ltf_count: u64,

    // GRBEnv. This variable is used for specifying the best LTF
    env: Env,
}


impl BddLtf {
    pub fn init(sample: &Sample<f64, f64>) -> BddLtf {
        let m = sample.len();

        // This is the default hash value for None node.
        let hash_for_none = (m + 1) as u64;


        let leaf = Rc::new(Node::new(m));
        let hash = leaf.borrow_mut().set_hash(hash_for_none);


        // Construct a hashmap that check the iisomorphism of two sub-tree.
        let mut map = HashMap::new();
        map.insert(hash, Rc::clone(&leaf));


        let root = Rc::new(
            Node::init(sample, &leaf, &mut map, hash_for_none)
        );


        let nodes = topological_sort(&root, &leaf);


        let ltf_count = root.borrow().count_ltfs(&leaf);


        let mut env = Env::new("").unwrap();
        env.set(param::OutputFlag, 0).unwrap();

        BddLtf { root, leaf, nodes, ltf_count, env }
    }


    pub fn count_ltfs(&self) -> u64 {
        let mut mp: HashMap<u64, u64> = HashMap::new();
        mp.insert(self.root.borrow().hash.unwrap(), 1_u64);

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
        // self.root.borrow().count_ltfs(&self.leaf)
    }


    fn init_path_info(&self) {
        for node in self.nodes.iter() {
            node.borrow_mut().path_info = PathInfo::new();
        }

        self.leaf.borrow_mut().path_info = PathInfo::new();

        self.root.borrow_mut().path_info.distance = 0.0;
    }


    pub fn summary(&self) {
        let n = self.nodes.len();
        println!(
            "# of LTFs: {}\n# of nodes: {}\n# of edges: {}",
            self.ltf_count, n + 1, 2 * n
        );
    }
}



/// This function computes a LTF that maximizes the edge.
/// Since `BddLtf` holds an NZDD that represents
/// all the representative hypotheses.
/// Each hypothesis in the NZDD corresponds to a labeling a LTF can realize.
/// 
/// 
/// First, we find a best prediction vector that maximizes the edge.
/// This step is computed by dynamic programming.
/// The complexity is $O(|V|)$.
/// 
/// After that, we compute a best LTF by solving a Quadratic programming.
/// The QP is the same as the hard-SVM.
/// By the definition of the NZDD, there is a feasible solution to the problem.
impl BaseLearner<f64, f64> for BddLtf {
    fn best_hypothesis(&self, sample: &Sample<f64, f64>, distribution: &[f64])
        -> Box<dyn Classifier<f64, f64>>
    {
        self.init_path_info();


        // Dynamic programming to find the longest path
        for (node_index, node) in self.nodes.iter().enumerate() {
            let distance = node.borrow().path_info.distance;
            let i = node.borrow().index;

            let d = distribution[i] * sample[i].label;

            if let Some(n) = &node.borrow_mut().neg {
                let _distance = distance - d;
                n.borrow_mut().path_info.assign(
                    _distance, Some(node_index), -1.0
                );
            }


            if let Some(p) = &node.borrow_mut().pos {
                let _distance = distance + d;
                p.borrow_mut().path_info.assign(
                    _distance, Some(node_index), 1.0
                );
            }
        }

        println!("Max edge: {}", self.leaf.borrow().path_info.distance);


        let m = sample.len();

        // Compute the best prediction vector
        let mut label = vec![0.0; m];
        let mut node  = &self.leaf;
        while let Some(index) = &node.borrow().path_info.prev_index {
            let source = &self.nodes[*index];
            let i = source.borrow().index;

            label[i] = node.borrow().path_info.pred_label;

            node = source;
        }


        // Since the label is computed by the backtracking,
        // there exists a hyperplane that realizes this labeling.
        // We find a normal vector by solving hard-SVM problem.
        let mut model = Model::with_env("BddLtf", &self.env)
            .unwrap();

        let dim = sample.feature_len();
        let normal = (0..dim).map(|i| {
            let name = format!("w[{}]", i);
            add_ctsvar!(model, name: &name, bounds: ..).unwrap()
        }).collect::<Vec<_>>();

        let intercept = add_ctsvar!(model, name: &"intercept", bounds: ..)
            .unwrap();

        model.update().unwrap();


        // Set constraints
        for (ex, &l) in sample.iter().zip(label.iter()) {
            let expr = (dot_product(&normal, &ex.data) + intercept) * l;

            model.add_constr(&"", c!(expr >= 1.0)).unwrap();
        }
        model.update().unwrap();


        // Set objective function
        // let objective = 0.5 * normal.iter()
        //     .map(|&v| v * v)
        //     .grb_sum();
        let objective = 0.0;
        model.set_objective(objective, Minimize).unwrap();
        model.update().unwrap();

        model.write(&"model.lp").unwrap();


        // Solve the optimization problem
        model.optimize().unwrap();

        let status = model.status().unwrap();
        if status != Status::Optimal && status != Status::SubOptimal {
            println!("Status: {:?}", status);
            panic!("Failed to finding a best ltf");
        }


        // Get the optimal solution
        let normal = normal.into_iter()
            .map(|n| model.get_obj_attr(attr::X, &n).unwrap())
            .collect::<Vec<f64>>();

        let intercept = model.get_obj_attr(attr::X, &intercept)
            .unwrap();

        // Construct a `LtfClassifier`
        let ltf = LtfClassifier::new(normal, intercept);

        Box::new(ltf)
    }
}


#[inline]
fn dot_product(vector: &[Var], data: &Data<f64>) -> Expr {
    match data {
        Data::Dense(dat) => {
            dat.iter()
                .zip(vector.iter())
                .map(|(&d, &v)| d * v)
                .grb_sum()
        },
        Data::Sparse(dat) => {
            dat.iter()
                .map(|(&i, &d)| d * vector[i])
                .grb_sum()
        }
    }
}


