use lycaon::base_learner::Classifier;
use lycaon::data_type::{Data, Label};


#[derive(Debug, Clone)]
pub struct SparseLtfClassifier {
    sparse_normal: Vec<(usize, f64)>,
    bias:          f64
}


impl SparseLtfClassifier {
    pub fn new(sparse_normal: Vec<(usize, f64)>, bias:f64)
        -> SparseLtfClassifier
    {
        SparseLtfClassifier { sparse_normal, bias }
    }
}


impl Classifier<f64, f64> for SparseLtfClassifier {
    fn predict(&self, data: &Data<f64>) -> Label<f64> {
        let val = self.sparse_normal.iter()
            .fold(0.0, |acc, (i, v)| acc + data.value_at(*i) * *v)
            + self.bias;

        if val > 0.0 {
             1.0
        } else {
            -1.0
        }
    }
}




