# Ultralytics YOLO 🚀, AGPL-3.0 license
# Updated for ultralytics >= 8.1 API compatibility

from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.models.yolo.model import YOLO
from ultralytics.utils import DEFAULT_CFG, LOGGER, ROOT
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import model_info, smart_inference_mode

from .predict import FastSAMPredictor


class FastSAM(YOLO):
    @smart_inference_mode()
    def predict(self, source=None, stream=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.engine.results.Results]): The prediction results.
        """
        if source is None:
            source = 'https://ultralytics.com/images/bus.jpg'
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
        overrides = self.overrides.copy()
        overrides['conf'] = 0.25
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = kwargs.get('mode', 'predict')
        assert overrides['mode'] in ['track', 'predict']
        overrides['save'] = kwargs.get('save', False)  # do not save by default if called in Python
        overrides['verbose'] = False  # suppress per-inference progress output
        self.predictor = FastSAMPredictor(overrides=overrides)
        self.predictor.setup_model(model=self.model, verbose=False)

        return self.predictor(source, stream=stream)

    def train(self, mode=True, **kwargs):
        """
        Keep nn.Module train/eval compatibility while blocking training API.

        `SAM_CD.eval()` recursively calls `child.train(False)`. FastSAM is used as a
        submodule there, so this method must accept the boolean `mode` argument.
        """
        if isinstance(mode, bool) and len(kwargs) == 0:
            if hasattr(self, 'model') and hasattr(self.model, 'train'):
                self.model.train(mode)
            return self
        raise NotImplementedError("FastSAM models don't support training")

    def val(self, **kwargs):
        """Run validation given dataset."""
        overrides = dict(task='segment', mode='val')
        overrides.update(kwargs)  # prefer kwargs
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)
        validator = FastSAM(args=args)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    @smart_inference_mode()
    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
        """
        overrides = dict(task='detect')
        overrides.update(kwargs)
        overrides['mode'] = 'export'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz:
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        if args.batch == DEFAULT_CFG.batch:
            args.batch = 1  # default to 1 if not modified
        return Exporter(overrides=args)(model=self.model)

    def info(self, detailed=False, verbose=True):
        """
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640)

    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection."""
        return self.predict(source, stream, **kwargs)

    def __getattr__(self, attr):
        """
        Defer missing attribute lookup to Ultralytics base implementation.

        SAM-CD relies on internals from ultralytics' Model/YOLO wrappers during
        initialization (e.g. `self.model`). Overriding this hook with an
        unconditional AttributeError breaks that flow on newer versions.
        """
        return super().__getattr__(attr)
