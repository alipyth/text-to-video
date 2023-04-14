!pip install modelscope==1.4.2
!pip install open_clip_torch
!pip install pytorch-lightning

from huggingface_hub import snapshot_download

from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import pathlib

model_dir = pathlib.Path('weights')
snapshot_download('damo-vilab/modelscope-damo-text-to-video-synthesis',
                   repo_type='model', local_dir=model_dir)

pipe = pipeline('text-to-video-synthesis', model_dir.as_posix())
test_text = {
        'text': 'elon musk riding a horse on the mars',
    }
output_video_path = pipe(test_text,)[OutputKeys.OUTPUT_VIDEO]
print('output_video_path:', output_video_path)
