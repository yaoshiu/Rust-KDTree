use numpy::ndarray::{s, stack, Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct KDTree {
    left: Option<Box<KDTree>>,
    point: Array1<i64>,
    right: Option<Box<KDTree>>,
}

#[derive(Debug)]
struct PointDis {
    distance: f64,
    point: Array1<i64>,
}

impl PartialEq for PointDis {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for PointDis {}

impl PartialOrd for PointDis {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PointDis {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl KDTree {
    pub fn new(points: ArrayView2<i64>) -> Option<Box<KDTree>> {
        let len = points.shape()[0];
        Self::build(&mut points.to_owned(), 0, len, 0)
    }

    fn build(
        points: &mut Array2<i64>,
        begin: usize,
        end: usize,
        axis: usize,
    ) -> Option<Box<KDTree>> {
        let mid = begin + end >> 1;

        let median = Self::nth_element(points, begin, end, axis, mid);

        let axis = (axis + 1) % median.len();

        let left = if mid - begin <= 0 {
            None
        } else {
            Self::build(points, begin, mid, axis)
        };
        let right = if end - mid - 1 <= 0 {
            None
        } else {
            Self::build(points, mid + 1, end, axis)
        };

        Some(Box::new(Self {
            left,
            right,
            point: median,
        }))
    }

    fn nth_element(
        points: &mut Array2<i64>,
        begin: usize,
        end: usize,
        axis: usize,
        n: usize,
    ) -> Array1<i64> {
        if end - begin <= 1 {
            return points.slice(s![begin, ..]).to_owned();
        }

        let pivot = points.slice(s![begin + end >> 1, ..]).to_owned().into_dyn();
        let (mut i, mut j, mut k) = (begin, begin, end);

        while i < k {
            if points[[i, axis]] < pivot[axis] {
                let j_sub = points.slice(s![j, ..]).to_owned();
                let i_sub = points.slice(s![i, ..]).to_owned();
                points.slice_mut(s![i, ..]).assign(&j_sub);
                points.slice_mut(s![j, ..]).assign(&i_sub);

                i += 1;
                j += 1;
            } else if points[[i, axis]] > pivot[axis] {
                k -= 1;

                let k_sub = points.slice(s![k, ..]).to_owned();
                let i_sub = points.slice(s![i, ..]).to_owned();
                points.slice_mut(s![i, ..]).assign(&k_sub);
                points.slice_mut(s![k, ..]).assign(&i_sub);
            } else {
                i += 1;
            }
        }

        if j > n {
            Self::nth_element(points, begin, j, axis, n)
        } else if n >= k {
            Self::nth_element(points, k, end, axis, n)
        } else {
            points.slice(s![n, ..]).to_owned()
        }
    }

    pub fn k_nearest(&self, query: ArrayView1<i64>, k: usize) -> Array2<i64> {
        let mut visited: HashSet<Self> = HashSet::new();
        let mut buf: BinaryHeap<PointDis> = BinaryHeap::new();
        self.knn_helper(&query, k, &mut visited, &mut buf, 0);
        stack(
            Axis(0),
            buf.iter()
                .map(|pd| pd.point.view())
                .collect::<Vec<ArrayView1<i64>>>()
                .as_slice(),
        )
        .unwrap()
    }

    fn knn_helper(
        &self,
        query: &ArrayView1<i64>,
        k: usize,
        visited: &mut HashSet<Self>,
        result: &mut BinaryHeap<PointDis>,
        axis: usize,
    ) {
        let distance = Self::get_distance(&self.point.view(), &query.view());
        let len = self.point.len();
        if self.left.is_none() && self.right.is_none() {
            if visited.contains(self) {
                return;
            }
            visited.insert(self.clone());
            result.push(PointDis {
                point: self.point.clone(),
                distance,
            });
            if result.len() > k {
                result.pop();
            }
        } else {
            let axis_dis = query[axis] - self.point[axis];
            if axis_dis < 0 {
                if let Some(child) = self.left.as_ref() {
                    child.knn_helper(query, k, visited, result, (axis + 1) % len);
                }
            } else if let Some(child) = self.right.as_ref() {
                child.knn_helper(query, k, visited, result, (axis + 1) % len);
            }

            if visited.contains(self) {
                return;
            }
            visited.insert(self.clone());

            result.push(PointDis {
                point: self.point.clone(),
                distance,
            });
            if result.len() > k {
                result.pop();
            }

            if result.peek().unwrap().distance > axis_dis.abs() as f64 || result.len() < k {
                if axis_dis < 0 {
                    if let Some(child) = self.right.as_ref() {
                        child.knn_helper(query, k, visited, result, (axis + 1) % len);
                    }
                } else if let Some(child) = self.left.as_ref() {
                    child.knn_helper(query, k, visited, result, (axis + 1) % len);
                }
            }
        }
    }

    fn get_distance(x: &ArrayView1<i64>, y: &ArrayView1<i64>) -> f64 {
        x.iter()
            .zip(y.iter())
            .fold(0.0, |acc, (&a, &b)| acc + ((a - b) * (a - b)) as f64)
            .sqrt()
    }
}
