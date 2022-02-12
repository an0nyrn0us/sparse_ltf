use std::convert::From;
use core::f64::consts::PI;

const HALFPI: f64 = PI / 2.0;

#[derive(Debug)]
pub struct Cartesian {
    pub x: f64,
    pub y: f64
}

impl Cartesian {
    pub fn new(x: f64, y: f64) -> Self {
        Cartesian { x, y }
    }
}


impl From<Polar> for Cartesian {
    fn from(p: Polar) -> Self {
        let x = p.r * p.theta.cos();
        let y = p.r * p.theta.sin();

        Cartesian { x, y }
    }
}




#[derive(Debug)]
pub struct Polar {
    pub r: f64,
    pub theta: f64
}


impl From<Cartesian> for Polar {
    fn from(p: Cartesian) -> Self {
        let r = (p.x.powf(2.0) + p.y.powf(2.0)).sqrt();

        let theta = (p.y / p.x).atan();

        Polar { r, theta }
    }
}


#[derive(Debug)]
pub struct Vertex {
    pub small: f64,
    pub large: f64
}


impl Vertex {
    pub fn new(small: f64, large: f64) -> Self {
        Vertex { small, large }
    }
}


fn add_vertices(vec: &mut Vec<Vertex>, v: Vertex, theta: f64) {
    assert!(
        v.small < theta && theta < v.large
    );

    vec.push(
        Vertex::new(v.small, theta)
    );

    vec.push(
        Vertex::new(theta, v.large)
    );
}


pub fn count_planes(points: Vec<Cartesian>) -> usize {

    let mut points = points
        .into_iter()
        .map(|p| p.into())
        .collect::<Vec<Polar>>()
        .into_iter();


    let mut current_level = Vec::new();
    if let Some(p) = points.next() {
        let l = p.theta - HALFPI;
        let r = p.theta + HALFPI;

        current_level.push(
            Vertex::new(l, r)
        );

        let l = l + PI;
        let r = r + PI;

        current_level.push(
            Vertex::new(l, r)
        );
    }



    while let Some(p) = points.next() {
        let pl = p.theta - HALFPI;
        let pr = p.theta + HALFPI;

        let ql = pl + PI;
        let qr = pr + PI;

        // println!("current level: {:?}", current_level);

        let mut next_level = Vec::new();

        for v in current_level {
            if v.small < pl && pl < v.large {
                add_vertices(&mut next_level, v, pl);
            } else if v.small < pr && pr < v.large {
                add_vertices(&mut next_level, v, pr);
            } else if v.small < ql && ql < v.large {
                add_vertices(&mut next_level, v, ql);
            } else if v.small < qr && qr < v.large {
                add_vertices(&mut next_level, v, qr);
            } else {
                next_level.push(v);
            }
        }
        current_level = next_level;
    }
    // println!("current level: {:?}", current_level);


    current_level.len()
}



pub fn normal_vectors(points: Vec<Cartesian>) -> Vec<Cartesian> {

    let points = points
        .into_iter()
        .map(|p| p.into())
        .collect::<Vec<Polar>>();


    let mut points = points.into_iter();


    let mut current_level = Vec::new();
    if let Some(p) = points.next() {
        let l = (p.theta - HALFPI) % (2.0 * PI);
        let r = (p.theta + HALFPI) % (2.0 * PI);

        current_level.push(
            Vertex::new(l, r)
        );

        let l = (p.theta +       PI / 2.0) % (2.0 * PI);
        let r = (p.theta + 3.0 * PI / 2.0) % (2.0 * PI);

        current_level.push(
            Vertex::new(l, r)
        );
    }



    while let Some(p) = points.next() {
        let pl = (p.theta - HALFPI) % (2.0 * PI);
        let pr = (p.theta + HALFPI) % (2.0 * PI);

        let mut next_level = Vec::new();

        for v in current_level {
            assert!(
                v.large - v.small <= 2.0 * PI
            );
            if v.small < pl && pl < v.large {
                add_vertices(&mut next_level, v, pl);
            } else if v.small < pr && pr < v.large {
                add_vertices(&mut next_level, v, pr);
            } else {
                next_level.push(v);
            }
        }
        current_level = next_level;
    }

    let mut normal_vectors = Vec::new();

    for v in current_level {
        let p = Polar {
            r: 1.0,
            theta: (v.large + v.small) / 2.0
        };

        normal_vectors.push(
            p.into()
        )
    }

    normal_vectors

}
