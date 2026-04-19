"""Main application configuration with dependency injection."""

from dataclasses import dataclass
from pathlib import Path

from .file_config import FileConfig
from .model_config import ModelConfig
from .ui_config import UIConfig


@dataclass
class AppConfig:
    """Unified application configuration with lazy-loaded sub-configs."""

    base_dir: Path

    def __post_init__(self) -> None:
        """Validate base_dir."""
        if not isinstance(self.base_dir, Path):
            self.base_dir = Path(self.base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {self.base_dir}")

    @property
    def file_config(self) -> FileConfig:
        """Get file path configuration."""
        return FileConfig(base_dir=self.base_dir)

    @property
    def model_config(self) -> ModelConfig:
        """Get model training configuration."""
        return ModelConfig()

    @property
    def ui_config(self) -> UIConfig:
        """Get UI configuration."""
        return UIConfig()

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create AppConfig from environment or defaults.

        Looks for app root as parent of 'src' directory.
        """
        # Find base directory (parent of src/)
        current = Path.cwd()
        while current != current.parent:
            if (current / "src").exists():
                base_dir = current
                break
            current = current.parent
        else:
            # Fallback to current directory
            base_dir = Path.cwd()

        return cls(base_dir=base_dir)
