use lycaon::base_learner::Classifier;
use lycaon::data_type::{Data, Label};


#[derive(Debug, Clone)]
pub struct LtfClassifier {
    normal:    Vec<f64>,
    intercept: f64
}


impl LtfClassifier {
    pub fn new(normal: Vec<f64>, intercept: f64) -> LtfClassifier {
        LtfClassifier { normal, intercept }
    }
}


impl Classifier<f64, f64> for LtfClassifier {
    fn predict(&self, data: &Data<f64>) -> Label<f64> {
        let dot = match data {
            Data::Dense(dat)  => {
                self.normal.iter()
                    .zip(dat.iter())
                    .fold(0.0, |acc, (&n, &d)| acc + n * d) + self.intercept
            },
            Data::Sparse(dat) => {
                dat.iter()
                    .fold(0.0, |acc, (i, d)| {
                        let n = self.normal[*i];
                        // let n = unsafe { self.normal.get_unchecked(i) };
                        acc + n * d
                    }) + self.intercept
            }
        };

        dot.signum()
    }
}




