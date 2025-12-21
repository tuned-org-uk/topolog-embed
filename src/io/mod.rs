use std::path::Path;

use anyhow::anyhow;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use genegraph_storage::StorageError;
// Corrected import
use genegraph_storage::lance_storage_graph::LanceStorageGraph;
use genegraph_storage::traits::backend::StorageBackend;
use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;

/// Load a DenseMatrix from storage and convert to Burn Tensor (N, D).
pub async fn load_dense_as_tensor<B: Backend, S: StorageBackend>(
    storage: &S,
    key: &str,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 2>> {
    let matrix: DenseMatrix<f64> = storage.load_dense(key).await?;
    let (nrows, ncols) = matrix.shape();

    let data_f32: Vec<f32> = matrix.iter().map(|x| *x as f32).collect();

    let tensor_data = TensorData::new(data_f32, vec![nrows, ncols]);
    Ok(Tensor::from_data(tensor_data, device))
}

/// Save a Burn Tensor (N, D) back to storage as DenseMatrix.
pub async fn save_tensor_as_dense<B: Backend, S: StorageBackend>(
    storage: &S,
    key: &str,
    tensor: Tensor<B, 2>,
    md_path: &std::path::Path,
) -> anyhow::Result<()> {
    let [nrows, ncols] = tensor.dims();
    let tensor_data = tensor.into_data(); // .into_data() is the new API
    let vec_f32 = tensor_data.to_vec::<f32>().unwrap();
    let vec_f64: Vec<f64> = vec_f32.iter().map(|&x| x as f64).collect();

    let matrix = DenseMatrix::from_iterator(vec_f64.into_iter(), nrows, ncols, 0);
    storage.save_dense(key, &matrix, md_path).await?;
    Ok(())
}

/// Load from Vec<Vec<f64>> into a Burn Tensor.
pub fn load_from_vec<B: Backend>(
    data: Vec<Vec<f64>>,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 2>> {
    if data.is_empty() || data[0].is_empty() {
        return Err(anyhow!("Input vector cannot be empty."));
    }

    let nrows = data.len();
    let ncols = data[0].len();

    let flat_data: Vec<f32> = data
        .into_iter()
        .flat_map(|row| row.into_iter().map(|val| val as f32))
        .collect();

    if flat_data.len() != nrows * ncols {
        return Err(anyhow!("Inconsistent row lengths in input vector."));
    }

    let tensor_data = TensorData::new(flat_data, vec![nrows, ncols]);
    Ok(Tensor::from_data(tensor_data, device))
}

/// Load a dense matrix from a .lance or .parquet file and convert to a Burn Tensor.
/// This function fulfills the `load_from_file` requirement.
pub async fn load_from_file<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<Tensor<B, 2>, StorageError> {
    // Create a temporary storage object to call the load_dense_from_file method,
    // which is part of the genegraph-storage API. [file:1]
    let parent_dir = path
        .parent()
        .ok_or_else(|| StorageError::Io(format!("Path has no parent: {:?}", path)))?
        .to_str()
        .ok_or_else(|| StorageError::Io(format!("Invalid parent path for {:?}", path)))?
        .to_string();

    // A dummy name is sufficient as the method only uses the path argument.
    let tmp_storage = LanceStorageGraph::new(parent_dir, "tmp_loader".to_string());

    // Call the async method from genegraph-storage. [file:1]
    let matrix: DenseMatrix<f64> = tmp_storage.load_dense_from_file(path).await?;

    // Convert the column-major DenseMatrix from smartcore to a row-major Vec for Burn.
    let (nrows, ncols) = matrix.shape();
    let mut row_major_data = Vec::with_capacity(nrows * ncols);
    for r in 0..nrows {
        for c in 0..ncols {
            row_major_data.push(*matrix.get((r, c)) as f32);
        }
    }

    // Create the tensor from the row-major data.
    let tensor_data = TensorData::new(row_major_data, vec![nrows, ncols]);
    Ok(Tensor::from_data(tensor_data, device))
}
