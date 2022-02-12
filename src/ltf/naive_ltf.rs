use lycaon::base_learner::{BaseLearner, Classifier};
use lycaon::data_type::{Sample, Data};

use grb::prelude::*;

use super::ltf_classifier::*;


pub struct NaiveLtf {
    env: Env,
}


impl NaiveLtf {
    pub fn init(_sample: &Sample<f64, f64>) -> NaiveLtf {
        let mut env = Env::new("").unwrap();
        env.set(param::OutputFlag, 0).unwrap();

        NaiveLtf { env }
    }
}


/// Find a **nice** hypothesis that nearly maximizes the edge.
/// This function does not guarantee to return a best hypothesis.
/// This implementation is based on the eq. (13) of following paper:
/// https://elkingarcia.github.io/Papers/MLDM07.pdf
impl BaseLearner<f64, f64> for NaiveLtf {
    fn best_hypothesis(&self, sample: &Sample<f64, f64>, distribution: &[f64])
        -> Box<dyn Classifier<f64, f64>>
    {
        let m   = sample.len();
        let dim = sample.feature_len();


        // The value C of C-SVM
        let hyper_param = 1e6;


        let mut model = Model::with_env("NaiveLtf", &self.env)
            .unwrap();


        let normal = (0..dim).into_iter()
            .map(|i| {
                let name = format!("w[{}]", i);
                add_ctsvar!(model, name: &name, bounds: ..).unwrap()
            }).collect::<Vec<_>>();

        let intercept = add_ctsvar!(model, name: &"intercept", bounds: ..)
            .unwrap();

        let xi_vec = (0..m).into_iter()
            .map(|i| {
                let name = format!("xi[{}]", i);
                add_ctsvar!(model, name: &name, bounds: 0.0..).unwrap()
            }).collect::<Vec<_>>();


        for (ex, &xi) in sample.iter().zip(xi_vec.iter()) {
            let expr = ex.label
                * (dot_product(&normal, &ex.data) + intercept);

            model.add_constr(&"", c!(expr >= 1.0 - xi)).unwrap();
        }
        model.update().unwrap();


        let objective = {
            let quad = normal.iter()
                .map(|&v| 0.5 * (v * v))
                .grb_sum();

            // Weighted hinge loss
            let lin = xi_vec.iter()
                .zip(distribution.iter())
                .map(|(&xi, &d)| xi * d)
                .grb_sum();

            quad + hyper_param * lin
        };
        model.set_objective(objective, Minimize).unwrap();
        model.update().unwrap();

        model.optimize().unwrap();


        let status = model.status().unwrap();
        if status != Status::Optimal && status != Status::SubOptimal {
            println!("Status: {:?}", status);
            panic!("Failed to finding a best ltf");
        }


        // Get the optimal solution
        let normal = normal.iter()
            .map(|n| model.get_obj_attr(attr::X, &n).unwrap())
            .collect::<Vec<f64>>();

        let intercept = model.get_obj_attr(attr::X, &intercept)
            .unwrap();

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



