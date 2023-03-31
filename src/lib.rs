use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::prelude::*;

mod kdt;

#[pyclass]
struct KDTree {
    kdt: Box<kdt::KDTree>,
}

#[pymethods]
impl KDTree {
    #[new]
    fn new(mut train: PyReadwriteArrayDyn<i64>) -> PyResult<Self> {
        Ok(Self {
            kdt: kdt::KDTree::new(train.as_array_mut()).unwrap(),
        })
    }

    fn k_nearest<'py>(&self, query: PyReadwriteArrayDyn<i64>, k: usize) -> Py<PyArrayDyn<i64>> {
        Python::with_gil(|py| {
            self.kdt
                .k_nearest(query.as_array(), k)
                .into_pyarray(py)
                .to_owned()
        })
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn knn(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "knn")]
    fn wrapped_knn<'py>(
        py: Python<'py>,
        mut train: PyReadwriteArrayDyn<i64>,
        query: PyReadonlyArrayDyn<i64>,
        k: usize,
    ) -> &'py PyArrayDyn<i64> {
        let kdt = kdt::KDTree::new(train.as_array_mut()).unwrap();
        kdt.k_nearest(query.as_array(), k).into_pyarray(py)
    }

    m.add_class::<KDTree>()?;

    Ok(())
}
