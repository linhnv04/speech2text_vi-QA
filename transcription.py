import os
import torch
import numpy as np
import librosa
from ruamel.yaml import YAML
import nemo
import nemo.collections.asr as nemo_asr
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import warnings
warnings.filterwarnings("ignore")


class AudioDataLayer(DataLayerNM):
    @property
    def output_ports(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
            torch.as_tensor(self.signal_shape, dtype=torch.int64)

    def set_signal(self, signal):
        self.signal = np.reshape(signal, [1, -1])
        self.signal_shape = np.expand_dims(self.signal.size, 0).astype(np.int64)
        self.output = True

    def __len__(self):
        return 1

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self


class VietASR:
    def __init__(self, config_file, encoder_checkpoint, decoder_checkpoint, lm_path=None, beam_width=20, lm_alpha=0.5, lm_beta=1.5):
        yaml = YAML(typ="safe")
        with open(config_file, encoding="utf-8") as f:
            model_definition = yaml.load(f)

        model_definition['AudioToMelSpectrogramPreprocessor']['dither'] = 0
        model_definition['AudioToMelSpectrogramPreprocessor']['pad_to'] = 0

        device = nemo.core.DeviceType.GPU if torch.cuda.is_available() else nemo.core.DeviceType.CPU
        neural_factory = nemo.core.NeuralModuleFactory(placement=device)

        data_layer = AudioDataLayer(sample_rate=model_definition['AudioToMelSpectrogramPreprocessor']['sample_rate'])
        data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(**model_definition['AudioToMelSpectrogramPreprocessor'])
        jasper_encoder = nemo_asr.JasperEncoder(feat_in=model_definition['AudioToMelSpectrogramPreprocessor']['features'], **model_definition['JasperEncoder'])
        jasper_decoder = nemo_asr.JasperDecoderForCTC(feat_in=model_definition['JasperEncoder']['jasper'][-1]['filters'], num_classes=len(model_definition['labels']))

        beamsearch_decoder = nemo_asr.BeamSearchDecoderWithLM(
            vocab=model_definition['labels'],
            beam_width=beam_width,
            alpha=lm_alpha,
            beta=lm_beta,
            lm_path=lm_path,
            num_cpus=max(1, os.cpu_count())
        )

        jasper_encoder.restore_from(encoder_checkpoint)
        jasper_decoder.restore_from(decoder_checkpoint)

        audio_signal, audio_signal_len = data_layer()
        processed_signal, processed_signal_len = data_preprocessor(input_signal=audio_signal, length=audio_signal_len)
        encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=processed_signal_len)
        log_probs = jasper_decoder(encoder_output=encoded)
        beam_predictions = beamsearch_decoder(log_probs=log_probs, log_probs_length=encoded_len)

        self.data_layer = data_layer
        self.neural_factory = neural_factory
        self.infer_tensors = [beam_predictions]

    def transcribe(self, audio_path):
        audio_signal, _ = librosa.load(audio_path, sr=16000)
        self.data_layer.set_signal(audio_signal)
        evaluated_tensors = self.neural_factory.infer(tensors=self.infer_tensors, verbose=False)
        return evaluated_tensors[0][0]


# path = "/home/alex/workspace/FPT_OJT/viet-asr/sample/1IJPK91LV_48BTAM.mp3"

config = 'configs/quartznet12x1_vi.yaml'
encoder_checkpoint = 'models/acoustic_model/vietnamese/JasperEncoder-STEP-289936.pt'
decoder_checkpoint = 'models/acoustic_model/vietnamese/JasperDecoderForCTC-STEP-289936.pt'
lm_path = 'models/language_model/3-gram-lm.binary'


asr = VietASR(
        config_file=config,
        encoder_checkpoint=encoder_checkpoint,
        decoder_checkpoint=decoder_checkpoint,
        lm_path=lm_path,
    )

def transcript(audio_path):
    return asr.transcribe(audio_path)



