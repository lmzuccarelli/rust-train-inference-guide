use crate::model::ModelConfig;
use crate::training::TrainingConfig;
use burn::data::dataloader::Dataset;
use burn::optim::AdamConfig;
use burn_autodiff::Autodiff;
use burn_cuda::{Cuda, CudaDevice};

mod data;
mod inference;
mod model;
mod training;

fn main() {
    let device = CudaDevice::default();
    type MyBackend = Cuda<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let artifact_dir = "guide";

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );

    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
