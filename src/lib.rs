use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

mod kdt;

#[pyclass]
struct KDTree {
    kdt: Box<kdt::KDTree>,
}

#[pymethods]
impl KDTree {
    #[new]
    fn new(train: PyReadonlyArray2<i64>) -> PyResult<Self> {
        Ok(Self {
            kdt: kdt::KDTree::new(train.as_array()).unwrap(),
        })
    }

    fn k_nearest<'py>(&self, query: PyReadonlyArray1<i64>, k: usize) -> Py<PyArray2<i64>> {
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
        train: PyReadonlyArray2<i64>,
        query: PyReadonlyArray1<i64>,
        k: usize,
    ) -> &'py PyArray2<i64> {
        kdt::KDTree::new(train.as_array())
            .unwrap()
            .k_nearest(query.as_array(), k)
            .into_pyarray(py)
    }

    m.add_class::<KDTree>()?;

    Ok(())
}
