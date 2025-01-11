import json
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf


device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

text_to_generate_voice = []

with open('input.json', 'r') as f:
    input_parameters = json.loads(f.read())
    for input_parameter in input_parameters:
        text = input_parameter.get('text')
        if text:
            text_to_generate_voice.append(text)

text_to_transcribe = '... '.join(text_to_generate_voice)

'''
TODO: Use numpy and soundfile to insert pauses between multiple output files
Presently, we use "..." to have Parler TTS insert pauses. However, we can add
precision to our pauses by defining the pause in our input.json (placeholder
key is "pause_in_seconds").
'''


description = '''
    A female speaker delivers a slightly expressive and animated speech with a
    moderate speed and pitch. The recording is of very high quality, with the 
    speaker's voice sounding clear and very close up."
'''


input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(
    text_to_transcribe, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("output/parler_tts_outp.wav", audio_arr, model.config.sampling_rate)