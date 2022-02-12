use lycaon::data_type::{Sample, Data};

use std::rc::Rc;
use std::cell::RefCell;
use std::hash::Hasher;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashSet, HashMap, VecDeque};


use grb::prelude::*;

#[derive(Debug)]
pub struct Node {
    pub index: usize,
    pub hash:  Option<u64>,
    pub pos:   Option<Rc<RefCell<Node>>>,
    pub neg:   Option<Rc<RefCell<Node>>>,

    pub(super) path_info: PathInfo,
}


impl Node {
    pub fn new(_index: usize) -> RefCell<Node> {
        RefCell::new(Node {
            index: _index,
            hash:  None,
            pos:   None,
            neg:   None,

            path_info: PathInfo::new()
        })
    }


    /// Update the `self.hash` and return it.
    pub fn set_hash(&mut self, hash_for_none: u64) -> u64 {
        let mut hasher = DefaultHasher::new();


        match &self.neg {
            None => {
                hasher.write_u64(hash_for_none);
            },
            Some(tree) => {
                match &tree.borrow().hash {
                    None => {
                        let _hash = tree.borrow_mut()
                            .set_hash(hash_for_none);
                        hasher.write_u64(_hash);
                    },
                    Some(_hash) => {
                        hasher.write_u64(*_hash);
                    }
                }
            }
        }


        hasher.write_u64(self.index as u64);


        match &self.pos {
            None => {
                hasher.write_u64(hash_for_none);
            },
            Some(tree) => {
                match &tree.borrow().hash {
                    None => {
                        let _hash = tree.borrow_mut()
                            .set_hash(hash_for_none);
                        hasher.write_u64(_hash);
                    },
                    Some(_hash) => {
                        hasher.write_u64(*_hash);
                    }
                }
            }
        }

        let hash  = hasher.finish();
        self.hash = Some(hash);

        hash
    }


    pub fn count_ltfs(&self, leaf: &Rc<RefCell<Node>>) -> u64 {
        let mut count = 0;

        if self.hash.unwrap() == leaf.borrow().hash.unwrap() {
            count += 1;
        } else {
            if let Some(p) = &self.pos {
                count += p.borrow().count_ltfs(leaf);
            }
            if let Some(n) = &self.neg {
                count += n.borrow().count_ltfs(leaf);
            }
        }

        count
    }


    /// This function constructs a BDD that represents the hyperplanes
    pub fn init(examples:      &Sample<f64, f64>,
                leaf:          &Rc<RefCell<Node>>,
                map:           &mut HashMap<u64, Rc<RefCell<Node>>>,
                hash_for_none: u64)
        -> RefCell<Node>
    {
        let dim = examples.feature_len();


        // Initialize the GRB Model
        let mut env = Env::new("").unwrap();

        env.set(param::OutputFlag, 0).unwrap();
        let mut model = Model::with_env("", env).unwrap();

        let mut ws = Vec::with_capacity(dim);
        for i in 0..dim {
            let name = format!("w[{}]", i);
            ws.push(
                add_ctsvar!(model, name: &name, bounds: ..).unwrap()
            );
        }

        let intercept = add_ctsvar!(model, name: &"intercept", bounds: ..)
            .unwrap();


        model.set_objective(0.0, Maximize).unwrap();

        model.update().unwrap();


        let root = Node::new(0);
        let lhs  = make_lhs(&ws, &examples[0].data) + intercept;
        let lhs2 = lhs.clone() * -1.0;


        let name   = "c0";
        let constr = model.add_constr(&name, c!(lhs >= 1.0))
            .unwrap();
        model.update().unwrap();
        root.borrow_mut().pos = construct_tree_inner(
            &mut model,
            &ws[..],
            &intercept,
            &examples,
            &leaf,
            1,
            map,
            hash_for_none
        );
        model.remove(constr).unwrap();
        model.update().unwrap();

        // println!("50% done");

        let constr = model.add_constr(&name, c!(lhs2 >= 1.0))
            .unwrap();
        model.update().unwrap();
        root.borrow_mut().neg = construct_tree_inner(
            &mut model,
            &ws[..],
            &intercept,
            &examples,
            &leaf,
            1,
            map,
            hash_for_none
        );
        model.remove(constr).unwrap();
        model.update().unwrap();

        root.borrow_mut().set_hash(hash_for_none);

        root
    }
}



fn construct_tree_inner(model:         &mut Model,
                        ws:            &[Var],
                        intercept:     &Var,
                        examples:      &Sample<f64, f64>,
                        leaf:          &Rc<RefCell<Node>>,
                        index:         usize,
                        map:           &mut HashMap<u64, Rc<RefCell<Node>>>,
                        hash_for_none: u64)
    -> Option<Rc<RefCell<Node>>>
{
    model.optimize().unwrap();
    let status = model.status().unwrap();

    // If the status is infeasible, we don't need to grow this tree.
    if status == Status::Infeasible {
        return None;
    }


    // If `index` is out of range, return the Rc.
    if index >= examples.len() {
        return Some(Rc::clone(leaf));
    }


    // Get the example at `index`.
    let ex = &examples[index].data;


    // Construct the tree.
    let tree = Node::new(index);


    // compute the inner product of `ex` and `ws`.
    let lhs  = make_lhs(ws, ex) + *intercept;
    let lhs2 = lhs.clone() * -1.0;
    let name = "";

    // Construct a sub-tree for the `neg` side.
    let constr = model.add_constr(&name, c!(lhs >= 1.0))
        .unwrap();
    model.update().unwrap();


    tree.borrow_mut().pos = construct_tree_inner(
        model,
        &ws[..],
        &intercept,
        &examples,
        &leaf,
        index + 1,
        map,
        hash_for_none
    );
    model.remove(constr).unwrap();
    model.update().unwrap();


    // Construct a sub-tree for the `neg` side.
    let constr = model.add_constr(&name, c!(lhs2 >= 1.0))
        .unwrap();
    model.update().unwrap();

    tree.borrow_mut().neg = construct_tree_inner(
        model,
        &ws[..],
        &intercept,
        &examples,
        &leaf,
        index + 1,
        map,
        hash_for_none
    );

    model.remove(constr).unwrap();
    model.update().unwrap();


    // Compute the hash value
    let hash = tree.borrow_mut().set_hash(hash_for_none);


    // Boxing
    let mut tree = Rc::new(tree);


    // Check whether the `tree` is already created or not.
    match map.get(&hash) {
        None => {
            map.insert(hash, Rc::clone(&tree));
        },
        Some(t) => {
            tree = Rc::clone(&t);
        }
    }


    Some(tree)
}


fn make_lhs(ws: &[Var], example: &Data<f64>) -> Expr {
    match example {
        Data::Sparse(ex) => {
            ex.iter().map(|(i, x)| *x * ws[*i]).grb_sum()
        },
        Data::Dense(ex) => {
            ex.iter().zip(ws.iter()).map(|(x, w)| *x * *w).grb_sum()
        }
    }
}


pub fn topological_sort(root: &Rc<RefCell<Node>>, leaf: &Rc<RefCell<Node>>)
    -> Vec<Rc<RefCell<Node>>>
{
    let mut que = VecDeque::new();
    let mut sorted = Vec::new();

    que.push_back(Rc::clone(&root));


    let mut set = HashSet::new();
    set.insert(root.borrow().hash);
    set.insert(leaf.borrow().hash);

    while let Some(node) = que.pop_front() {
        if let Some(l) = &node.borrow().neg {
            let h = l.borrow().hash;
            if !set.contains(&h) {
                set.insert(h);
                que.push_back(Rc::clone(&l));
            }
        }

        if let Some(r) = &node.borrow().pos {
            let h = r.borrow().hash;
            if !set.contains(&h) {
                set.insert(h);
                que.push_back(Rc::clone(&r));
            }
        }

        sorted.push(node);
    }

    // sorted.push(Rc::clone(&leaf));


    sorted
}




/// `PathInfo` stores the enough information to back-track the `BddLtf`
#[derive(Debug)]
pub(super) struct PathInfo {
    pub(super) distance:   f64,
    pub(super) prev_index: Option<usize>,
    pub(super) pred_label: f64,
}


impl PathInfo {
    pub(super) fn new() -> PathInfo {
        PathInfo {
            distance:   f64::MIN,
            prev_index: None,
            pred_label: 0.0
        }
    }


    pub(super) fn assign(&mut self,
                         distance: f64,
                         index: Option<usize>,
                         label: f64)
    {
        if self.distance < distance {
            self.distance   = distance;
            self.prev_index = index;
            self.pred_label = label;
        }
    }
}
