use std::path::PathBuf;

use burn::{
    prelude::Backend,
    store::{BurnpackStore, ModuleStore, PyTorchToBurnAdapter, SafetensorsStore},
};
use rootcause::{Report, prelude::ResultExt};
use thiserror::Error;

use crate::modules::{ae::AutoEncoder, transformer::ZImageModel};

#[derive(Error, Debug)]
pub enum ModelLoadError {
    #[error("Error while loading weights")]
    LoadError,
    #[error("Unrecognised file extension")]
    UnknownExtension,
}

impl<B: Backend> ZImageModel<B> {
    pub fn with_weights(
        mut self,
        path: impl Into<PathBuf>,
    ) -> Result<Self, Report<ModelLoadError>> {
        self.load_weights(path)?;
        Ok(self)
    }

    pub fn load_weights(&mut self, path: impl Into<PathBuf>) -> Result<(), Report<ModelLoadError>> {
        let path = path.into();
        let extension = path.extension().map(|s| s.to_string_lossy().to_lowercase());

        match extension.as_deref() {
            Some("safetensors") => {
                let mut weights = SafetensorsStore::from_file(path)
                    .with_from_adapter(PyTorchToBurnAdapter::default())
                    .with_key_remapping(r"adaLN_modulation\.1", "adaln_modulation")
                    .with_key_remapping(r"adaLN_modulation\.0", "adaln_modulation")
                    .with_key_remapping(r"cap_embedder\.", "cap_embedder_")
                    .with_key_remapping(r"cap_embedder_0.weight", "cap_embedder_0.gamma")
                    .with_key_remapping(r"_norm\.weight", "_norm.gamma")
                    .with_key_remapping(r"norm1\.weight", "norm1.gamma")
                    .with_key_remapping(r"norm2\.weight", "norm2.gamma")
                    .with_key_remapping(r"attention\.out", "attention.to_out")
                    .with_key_remapping(r"^t_embedder\.mlp\.0", "t_embedder.mlp_1")
                    .with_key_remapping(r"^t_embedder\.mlp\.2", "t_embedder.mlp_2");
                let _result = weights.apply_to(self).context(ModelLoadError::LoadError)?;
            }
            Some("bpk") | None => {
                let mut weights = BurnpackStore::from_file(path)
                    .auto_extension(false)
                    .zero_copy(true);
                let _result = weights.apply_to(self).context(ModelLoadError::LoadError)?;
            }
            _ => {
                return Err(Report::new(ModelLoadError::UnknownExtension));
            }
        }

        Ok(())
    }
}

impl<B: Backend> AutoEncoder<B> {
    pub fn with_weights(
        mut self,
        path: impl Into<PathBuf>,
    ) -> Result<Self, Report<ModelLoadError>> {
        self.load_weights(path)?;
        Ok(self)
    }

    pub fn load_weights(&mut self, path: impl Into<PathBuf>) -> Result<(), Report<ModelLoadError>> {
        let path = path.into();
        let extension = path.extension().map(|s| s.to_string_lossy().to_lowercase());

        match extension.as_deref() {
            Some("safetensors") => {
                let mut weights = SafetensorsStore::from_file(path)
                    .with_from_adapter(PyTorchToBurnAdapter::default())
                    .with_key_remapping(r"mid\.(.*)", "mid_$1");
                weights.apply_to(self).context(ModelLoadError::LoadError)?;
            }
            Some("bpk") | None => {
                let mut weights = BurnpackStore::from_file(path)
                    .auto_extension(false)
                    .zero_copy(true);
                weights.apply_to(self).context(ModelLoadError::LoadError)?;
            }
            _ => {
                return Err(Report::new(ModelLoadError::UnknownExtension));
            }
        }

        Ok(())
    }
}
