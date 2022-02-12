use lycaon::read_csv;
use lycaon::booster::{LPBoost, Booster};

use crate::ltf::BddLtf;
use crate::ltf::NaiveLtf;


#[cfg(test)]
pub mod ltf_test {
    use super::*;
    #[test]
    fn german() {
        let path = "/Users/ryotaromitsuboshi/Documents/Habataki/boosting/rust/src/shrinked_german.csv";

        let sample = read_csv(path).unwrap();

        let m   = sample.len();


        let mut lpboost = LPBoost::init(&sample).capping(0.2 * m as f64);


        let ltf = BddLtf::init(&sample);

        ltf.summary();


        lpboost.run(&ltf, &sample, 0.01);
        println!("\t\tOptimal value: {}", lpboost.gamma_hat);



        let mut lpboost = LPBoost::init(&sample).capping(0.2 * m as f64);
        let ltf = NaiveLtf::init(&sample);
        lpboost.run(&ltf, &sample, 0.01);
        println!("\t\tOptimal value: {}", lpboost.gamma_hat);
    }
}
