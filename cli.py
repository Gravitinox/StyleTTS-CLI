import os
import sys
import torch
import time
import yaml
import multiprocessing
import shutil
from datetime import datetime
import glob
import argparse
import random

from styletts2.utils import *
from modules.tortoise_dataset_tools.dataset_whisper_tools.dataset_maker_large_files import *
from modules.tortoise_dataset_tools.dataset_whisper_tools.combine_folders import *

# Path to the settings file
SETTINGS_FILE_PATH = "Configs/generate_settings.yaml"
GENERATE_SETTINGS = {}
TRAINING_DIR = "training"
BASE_CONFIG_FILE_PATH = r"Configs\template_config_ft.yml"
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]
VALID_AUDIO_EXT = [
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".m4a",
    ".opus"
]
language = "en"


device = 'cuda' if torch.cuda.is_available() else 'cpu'
global_phonemizer = None
model = None
model_params = None
sampler = None
textcleaner = None
to_mel = None

def load_all_models(model_path):
    global global_phonemizer, model, model_params, sampler, textcleaner, to_mel
    
    model_config = get_model_configuration(model_path)
    if not model_config:
        return None
    
    config = load_configurations(model_config)
    sigma_value = config['model_params']['diffusion']['dist']['sigma_data']
    
    model, model_params = load_models_webui(sigma_value, device)
    global_phonemizer = load_phonemizer()
    
    sampler = create_sampler(model)
    textcleaner = TextCleaner()
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    
    load_pretrained_model(model, model_path=model_path)
    return False


def get_file_path(root_path, voice, file_extension, error_message):
    model_path = os.path.join(root_path, voice)
    if not os.path.exists(model_path):
        raise gr.Error(f'No {file_extension} located in "{root_path}" folder')

    for file in os.listdir(model_path):
        if file.endswith(file_extension):
            return os.path.join(model_path, file)
    
    raise gr.Error(error_message)

def get_model_configuration(model_path):
    base_directory, _ = os.path.split(model_path)
    for file in os.listdir(base_directory):
        if file.endswith(".yml"):
            configuration_path = os.path.join(base_directory, file)
            return configuration_path
    
    raise gr.Error("No configuration file found in the model folder")
    
def load_voice_model(voice):
    return get_file_path(root_path="models", voice=voice, file_extension=".pth", error_message="No TTS model found in specified location")

def generate_audio(text, voice, reference_audio_file, seed, alpha, beta, diffusion_steps, embedding_scale, voice_model, voices_root="voices"):
    original_seed = int(seed)
    reference_audio_path = os.path.join(voices_root, voice, reference_audio_file)
    reference_dicts = {f'{voice}': f"{reference_audio_path}"}
    # noise = torch.randn(1, 1, 256).to(device)
    start = time.time()
    if original_seed == -1:
        seed_value = random.randint(0, 2**32 - 1)
    else:
        seed_value = original_seed
    set_seeds(seed_value)
    for k, path in reference_dicts.items():
        mean, std = -4, 4
        ref_s = compute_style(path, model, to_mel, mean, std, device)
        
        wav1 = inference(text, ref_s, model, sampler, textcleaner, to_mel, device, model_params, global_phonemizer=global_phonemizer, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)

        from scipy.io.wavfile import write
        os.makedirs("results", exist_ok=True)
        audio_opt_path = os.path.join("results", f"{voice}_output.wav")
        write(audio_opt_path, 24000, wav1)
        generate_elapsed_time = time.time() - start
        generated_audio_length = librosa.get_duration(path=audio_opt_path)
        rtf = generate_elapsed_time / generated_audio_length
        print(f"{k} Synthesized: {text}")
        print(f"Processing time: {generate_elapsed_time:.2f} seconds.")
        print(f"Generated audio length: {generated_audio_length:.2f} seconds.")
        print(f"RTF = {rtf:5f}\n")

    # Save the settings after generation
    save_settings({
        "text": text,
        "voice": voice,
        "reference_audio_file": reference_audio_file,
        "seed": original_seed if original_seed == -1 else seed_value,
        "alpha": alpha,
        "beta": beta,
        "diffusion_steps": diffusion_steps,
        "embedding_scale": embedding_scale,
        "voice_model" : voice_model
    })
    return audio_opt_path, seed_value

def train_model(data):
    return f"Model trained with data: {data}"

def update_settings(setting_value):
    return f"Settings updated to: {setting_value}"

def get_folder_list(root):
    folder_list = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
    return folder_list

def get_reference_audio_list(voice_name, root="voices"):
    reference_directory_list = os.listdir(os.path.join(root, voice_name))
    return reference_directory_list

def get_voice_models():
    folders_to_browse = ["training", "models"]
    
    model_list = []

    for folder in folders_to_browse:
        # Construct the search pattern
        search_pattern = os.path.join(folder, '**', '*.pth')
        # Use glob to find all matching files, recursively search in subfolders
        matching_files = glob.glob(search_pattern, recursive=True)
        # Extend the model_list with the found files
        model_list.extend(matching_files)
        
    return model_list
        
    
def update_reference_audio(voice):
    return gr.Dropdown(choices=get_reference_audio_list(voice), value=get_reference_audio_list(voice)[0])

def update_voice_model(model_path):
    gr.Info("Wait for models to load...")
    # model_path = get_models_path(voice, model_name)
    path_components = model_path.split(os.path.sep)
    voice = path_components[1]
    loaded_check = load_all_models(model_path=model_path)
    if loaded_check:
        raise gr.Warning("No model or model configuration loaded, check model config file is present")
    gr.Info("Models finished loading")

def get_models_path(voice, model_name, root="models"):
    return os.path.join(root, voice, model_name)

def update_voice_settings(voice):
    try:
        # gr.Info("Wait for models to load...")
        # model_name = get_voice_models(voice)    
        # model_path = get_models_path(voice, model_name[0])   
        # loaded_check = load_all_models(model_path=model_path)
        # if loaded_check == None:
        #     gr.Warning("No model or model configuration loaded, check model config file is present")
        ref_aud_path = update_reference_audio(voice)
        
        # gr.Info("Models finished loading")
        return ref_aud_path #gr.Dropdown(choices=model_name, value=model_name[0] if model_name else None)
    except:
        gr.Warning("No models found for the chosen voice chosen, new models not loaded")
        ref_aud_path = update_reference_audio(voice)
        return ref_aud_path, gr.Dropdown(choices=[]) 

def load_settings():
    try:
        with open(SETTINGS_FILE_PATH, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        if reference_audio_list:
            reference_file = reference_audio_list[0]
        else:
            reference_file = None
        if voice_list_with_defaults:
            voice = voice_list_with_defaults[0]
        else:
            voice = None
         
        settings_list = {
            "text": "Inferencing with this sentence, just to make sure things work!",
            "voice": voice,
            "reference_audio_file": reference_file,
            "seed" : "-1",
            "alpha": 0.3,
            "beta": 0.7,
            "diffusion_steps": 30,
            "embedding_scale": 1.0,
            "voice_model" : "models\pretrain_base_1\epochs_2nd_00020.pth"
        }
        return settings_list

def save_settings(settings):
    with open(SETTINGS_FILE_PATH, "w") as f:
        yaml.safe_dump(settings, f)
        
def update_button_proxy():
    voice_list_with_defaults = get_voice_list(append_defaults=True)
    datasets_list = get_voice_list(get_voice_dir("datasets"), append_defaults=True)
    train_list = get_folder_list(root="training")
    return gr.Dropdown(choices=voice_list_with_defaults), gr.Dropdown(choices=datasets_list), gr.Dropdown(choices=train_list), gr.Dropdown(choices=train_list)

def update_data_proxy(voice_name):
    train_data = os.path.join(TRAINING_DIR, voice_name,"train_phoneme.txt")
    val_data = os.path.join(TRAINING_DIR, voice_name, "validation_phoneme.txt")
    root_path = os.path.join(TRAINING_DIR, voice_name, "audio")
    return train_data, val_data, root_path

def save_yaml_config(config,  voice_name):
    os.makedirs(os.path.join(TRAINING_DIR, voice_name), exist_ok=True)  # Create the output directory if it doesn't exist
    output_file_path = os.path.join(TRAINING_DIR, voice_name, f"{voice_name}_config.yml")
    with open(output_file_path, 'w') as file:
        yaml.dump(config, file)

def get_dataset_continuation(voice):
    try:
        training_dir = f"training/{voice}/processed"
        if os.path.exists(training_dir):
            processed_dataset_list = [folder for folder in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, folder))]
            if processed_dataset_list:
                processed_dataset_list.append("")
                return gr.Dropdown(choices=processed_dataset_list, value="", interactive=True)
    except Exception as e:
        print(f"Error getting dataset continuation: {str(e)}")
    return gr.Dropdown(choices=[], value="", interactive=True)

def load_whisper_model(language=None, model_name=None, progress=None):
    import whisperx
    # import whisper
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        device_index = 0
        compute_type = "int8"
        # raise gr.Error("Non-Nvidia GPU detected, or CUDA not available")
    whisper_model = whisperx.load_model(model_name, device, device_index, compute_type, download_root="whisper_models")
    try:
        whisper_model = whisperx.load_model(model_name, device, download_root="whisper_models", compute_type="float16")
    except Exception as e: # for older GPUs
        print(f"Debugging info: {e}")
        whisper_model = whisperx.load_model(model_name, device, download_root="whisper_models", compute_type="int8")
    # whisper_align_model = whisperx.load_align_model(model_name="WAV2VEC2_ASR_LARGE_LV60K_960H" if language=="en" else None, language_code=language, device=device)
    print("Loaded Whisper model")
    return whisper_model

def get_training_folder(voice) -> str:
    '''
    voice(str) : voice to retrieve training folder from
    '''
    return f"./training/{voice}"

def transcribe_and_process(voice, language, chunk_size, continuation_directory, align, rename, num_processes, keep_originals, srt_multiprocessing, ext, speaker_id, whisper_model):
    print("Starting transcription and processing...")
    print("Loading Whisper model...")
    time_transcribe = time.time()
    whisper_model = load_whisper_model(language=language, model_name=whisper_model)
    print("Processing audio files...")
    num_processes = int(num_processes)
    training_folder = get_training_folder(voice)
    processed_folder = os.path.join(training_folder,"processed")
    dataset_dir = os.path.join(processed_folder, "run")
    print("Merging segments...")
    merge_dir = os.path.join(dataset_dir, "dataset/wav_splits")
    audio_dataset_path = os.path.join(merge_dir, 'audio')
    train_text_path = os.path.join(dataset_dir, 'dataset/train.txt')
    validation_text_path = os.path.join(dataset_dir, 'dataset/validation.txt')
    
    large_file_num_processes = int(num_processes/2) # Used for instances where larger files are being processed, as to not run out of RAM
    
    items_to_move = [audio_dataset_path, train_text_path, validation_text_path]
    
    for item in items_to_move:
        if os.path.exists(os.path.join(training_folder, os.path.basename(item))):
            raise gr.Error(f'Remove ~~train.txt ~~validation.txt ~~audio(folder) from "./training/{voice}" before trying to transcribe a new dataset. Or click the "Archive Existing" button')
            
    if continuation_directory:
        dataset_dir = os.path.join(processed_folder, continuation_directory)

    elif os.path.exists(dataset_dir):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_dataset_dir = os.path.join(processed_folder, f"run_{current_datetime}")
        os.rename(dataset_dir, new_dataset_dir)

    from modules.tortoise_dataset_tools.audio_conversion_tools.split_long_file import get_duration, process_folder
    chosen_directory = os.path.join("./datasets", voice)
    print("chosen_directory =", chosen_directory)
    items = [item for item in os.listdir(chosen_directory) if os.path.splitext(item)[1].lower() in VALID_AUDIO_EXT]
    
    # This is to prevent an error below when processing "non audio" files.  This will occur with other types, but .pth should
    # be the only other ones in the voices folder.
    #for file in items:
    #    if file.endswith(".pth"):
    #        items.remove(file)
    
    # In case of sudden restart, removes this intermediary file used for rename
    for file in items:
        if "file___" in file:
            os.remove(os.path.join(chosen_directory, file))
    
    file_durations = [get_duration(os.path.join(chosen_directory, item)) for item in items if os.path.isfile(os.path.join(chosen_directory, item))]
    print("Splitting long files")
    if any(duration > 3600*2 for duration in file_durations):
        process_folder(chosen_directory, large_file_num_processes)
    
    if not keep_originals:
        originals_pre_split_path = os.path.join(chosen_directory, "original_pre_split")
        try:
            shutil.rmtree(originals_pre_split_path)
        except:
            # There is no directory to delete
            pass
            
    print("Converting to MP3 files") # add tqdm later
    
    if ext=="mp3":
        import modules.tortoise_dataset_tools.audio_conversion_tools.convert_to_mp3 as c2mp3
        
        # Hacky way to get the functions working without changing where they output to...
        for item in os.listdir(chosen_directory):
            if os.path.isfile(os.path.join(chosen_directory, item)):
                original_dir = os.path.join(chosen_directory, "original_files")
                if not os.path.exists(original_dir):
                    os.makedirs(original_dir)
                item_path = os.path.join(chosen_directory, item)
                try:
                    shutil.move(item_path, original_dir)
                except:
                    os.remove(item_path)
        
        try:
            c2mp3.process_folder(original_dir, large_file_num_processes)
        except:
            raise gr.Error('No files found in the voice folder specified, make sure it is not empty.  If you interrupted the process, the files may be in the "original_files" folder')
        
        # Hacky way to move the files back into the main voice folder
        for item in os.listdir(os.path.join(original_dir, "converted")):
            item_path = os.path.join(original_dir, "converted", item)
            if os.path.isfile(item_path):
                try:
                    shutil.move(item_path, chosen_directory)
                except:
                    os.remove(item_path)
            
    if not keep_originals:
        originals_files = os.path.join(chosen_directory, "original_files")
        try:
            shutil.rmtree(originals_files)
        except:
            # There is no directory to delete
            pass

    print("Processing audio files")
    
    process_audio_files(base_directory=dataset_dir,
                        language=language,
                        audio_dir=chosen_directory,
                        chunk_size=chunk_size,
                        no_align=align,
                        rename_files=rename,
                        num_processes=num_processes,
                        whisper_model=whisper_model,
                        srt_multiprocessing=srt_multiprocessing,
                        ext=ext,
                        speaker_id=speaker_id,
                        sr_rate=24000
                        )
    print("Audio processing completed")

    print("Merging segments")
    merge_segments(merge_dir)
    print("Segment merging completed")

    try:
        for item in items_to_move:
            if os.path.exists(os.path.join(training_folder, os.path.basename(item))):
                print("Already exists")
            else:
                shutil.move(item, training_folder)
        shutil.rmtree(dataset_dir)
    except Exception as e:
        raise gr.Error(e)
        
    print("Transcription and processing completed successfully!")
    print(time.strftime("Transcription processing time: %H:%M:%S", time.gmtime(time.time() - time_transcribe)))

    return "Transcription and processing completed successfully!"

def phonemize_files(voice):
    print("Starting phonemization...")
    training_root = get_training_folder(voice)
    train_text_path = os.path.join(training_root, "train.txt")
    print("train_text_path:", train_text_path)
    file1 = open(train_text_path, "r")
    print(file1.readlines())
    print()
    file1.close()
    train_opt_path = os.path.join(training_root, "train_phoneme.txt")
    print("train_opt_path:", train_opt_path)
    validation_text_path = os.path.join(training_root, "validation.txt")
    print("validation_text_path:", validation_text_path)
    validation_opt_path = os.path.join(training_root, "validation_phoneme.txt")
    print("validation_opt_path:", validation_opt_path)
    # Hardcoded to "both" to stay consistent with the train_to_phoneme.py script and not having to modify it
    option = "both"
    
    from modules.styletts2_phonemizer.train_to_phoneme import process_file
    
    print("Train Phonemization Starting...")
    print(train_text_path, train_opt_path, option)
    process_file(train_text_path, train_opt_path, option)
    print("Validation Phonemization Starting...")
    print(validation_text_path, validation_opt_path, option)
    process_file(validation_text_path, validation_opt_path, option)
    
    return "Phonemization complete!"

def archive_dataset(voice):
    print(f"Archiving dataset for voice: {voice}")
    training_folder = get_training_folder(voice)
    archive_root = os.path.join(training_folder,"archived_data")
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_folder = os.path.join(archive_root,current_datetime)
    
    items_to_move = ["train.txt", "validation.txt", "audio", "train_phoneme.txt", "validation_phoneme.txt"]
    training_folder_contents = os.listdir(training_folder)

    if not any(item in training_folder_contents for item in items_to_move):
        print("No files to move")
        return
    
    for item in items_to_move:
        os.makedirs(archive_folder, exist_ok=True)
        move_item_path = os.path.join(training_folder, item)
        if os.path.exists(move_item_path):
            try:
                shutil.move(move_item_path, archive_folder)
                print(f"Moved {item} to archive folder")
            except Exception as e:
                print(f"Error moving {item}: {str(e)}")
    
    print('Finished archiving files to "archived_data" folder')

voice_list_with_defaults = get_voice_list(append_defaults=True)
datasets_list = get_voice_list(get_voice_dir("datasets"), append_defaults=True)
if voice_list_with_defaults:
    reference_audio_list = get_reference_audio_list(voice_list_with_defaults[0])
    train_list = get_folder_list(root="training")
else:
    reference_audio_list = None
    voice_list_with_default = None
    train_list = None

def update_config(voice_name, save_freq, log_interval, epochs, batch_size, max_len, pretrained_model, load_only_params, F0_path, ASR_config, ASR_path, PLBERT_dir, train_data, val_data, root_path, diff_epoch, joint_epoch):
    print("Updating configuration...")
    train_data, val_data, root_path = update_data_proxy(voice_name)
    with open(BASE_CONFIG_FILE_PATH, "r") as f:
        config = yaml.safe_load(f)

    config["log_dir"] = os.path.join(TRAINING_DIR, voice_name, "models")
    config["save_freq"] = save_freq
    config["log_interval"] = log_interval
    config["epochs"] = epochs
    config["batch_size"] = batch_size
    config["max_len"] = max_len
    config["pretrained_model"] = pretrained_model
    config["load_only_params"] = load_only_params
    config["F0_path"] = F0_path
    config["ASR_config"] = ASR_config
    config["ASR_path"] = ASR_path
    config["PLBERT_dir"] = PLBERT_dir
    config["data_params"]["train_data"] = train_data
    config["data_params"]["val_data"] = val_data
    config["data_params"]["root_path"] = root_path
    config["loss_params"]["diff_epoch"] = diff_epoch
    config["loss_params"]["joint_epoch"] = joint_epoch

    save_yaml_config(config, voice_name=voice_name)
    return "Configuration updated successfully."

def start_training(voice):
    print(f"Starting training for voice: {voice}")
    from styletts2.train_finetune_accelerate import main as run_train
    config_path = os.path.join("training", voice, f"{voice}_config.yml")
    print(config_path)
    run_train(config_path)
    return "Training Complete!"

def main():
    parser = argparse.ArgumentParser(description="StyleTTS2 CLI")
    
    # General arguments
    parser.add_argument("--mode", choices=["generate", "transcribe", "phonemize", "archive", "update_config", "train"], required=True, help="Mode of operation")
    
    # Generate mode arguments
    generate_group = parser.add_argument_group("Generate mode arguments")
    generate_group.add_argument("--text", help="Input text for generation")
    generate_group.add_argument("--voice", help="Voice to use")
    generate_group.add_argument("--reference_audio", help="Reference audio file")
    generate_group.add_argument("--seed", type=int, default=-1, help="Seed for generation")
    generate_group.add_argument("--alpha", type=float, default=0.3, help="Alpha value")
    generate_group.add_argument("--beta", type=float, default=0.7, help="Beta value")
    generate_group.add_argument("--diffusion_steps", type=int, default=30, help="Number of diffusion steps")
    generate_group.add_argument("--embedding_scale", type=float, default=1.0, help="Embedding scale")
    generate_group.add_argument("--voice_model", help="Path to voice model")
    
    # Transcribe mode arguments
    transcribe_group = parser.add_argument_group("Transcribe mode arguments")
    transcribe_group.add_argument("--language", default="en", help="Language for transcription")
    transcribe_group.add_argument("--chunk_size", type=int, default=15, help="Chunk size for transcription")
    transcribe_group.add_argument("--continuation_directory", help="Continuation directory")
    transcribe_group.add_argument("--align", action="store_true", help="Disable WhisperX Alignment")
    transcribe_group.add_argument("--rename", action="store_true", help="Rename audio files")
    transcribe_group.add_argument("--num_processes", type=int, default=multiprocessing.cpu_count()-2, help="Number of processes to use")
    transcribe_group.add_argument("--keep_originals", action="store_true", help="Keep original files")
    transcribe_group.add_argument("--srt_multiprocessing", action="store_true", help="Enable SRT multiprocessing")
    transcribe_group.add_argument("--ext", choices=["wav", "mp3"], default="wav", help="Audio file extension")
    transcribe_group.add_argument("--speaker_id", action="store_true", help="Enable speaker ID")
    transcribe_group.add_argument("--whisper_model", choices=WHISPER_MODELS, default="large-v3", help="WhisperX model to use")
    
    # Update config mode arguments
    config_group = parser.add_argument_group("Update config mode arguments")
    config_group.add_argument("--save_freq", type=int, default=1, help="Save frequency")
    config_group.add_argument("--log_interval", type=int, default=10, help="Log interval")
    config_group.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    config_group.add_argument("--batch_size", type=int, default=2, help="Batch size")
    config_group.add_argument("--max_len", type=int, default=250, help="Max length")
    config_group.add_argument("--pretrained_model", default="models/pretrain_base_1/epochs_2nd_00020.pth", help="Path to pretrained model")
    config_group.add_argument("--load_only_params", type=bool, default=True, help="Load only parameters")
    config_group.add_argument("--F0_path", default="Utils/JDC/bst.t7", help="Path to F0 model")
    config_group.add_argument("--ASR_config", default="Utils/ASR/config.yml", help="Path to ASR config")
    config_group.add_argument("--ASR_path", default="Utils/ASR/epoch_00080.pth", help="Path to ASR model")
    config_group.add_argument("--PLBERT_dir", default="Utils/PLBERT", help="Path to PLBERT directory")
    config_group.add_argument("--train_data", help="Path to train data")
    config_group.add_argument("--val_data", help="Path to validation data")
    config_group.add_argument("--root_path", help="Root path for audio data")
    config_group.add_argument("--diff_epoch", type=int, default=0, help="Diffusion epoch")
    config_group.add_argument("--joint_epoch", type=int, default=0, help="Joint epoch")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        if not all([args.text, args.voice, args.reference_audio, args.voice_model]):
            parser.error("Generate mode requires --text, --voice, --reference_audio, and --voice_model")
        load_all_models(args.voice_model)
        audio_path, seed = generate_audio(args.text, args.voice, args.reference_audio, args.seed, args.alpha, args.beta, args.diffusion_steps, args.embedding_scale, args.voice_model)
        print(f"Generated audio saved to: {audio_path}")
        print(f"Seed used: {seed}")
    
    elif args.mode == "transcribe":
        if not args.voice:
            parser.error("Transcribe mode requires --voice")
        result = transcribe_and_process(args.voice, args.language, args.chunk_size, args.continuation_directory, args.align, args.rename, args.num_processes, args.keep_originals, args.srt_multiprocessing, args.ext, args.speaker_id, args.whisper_model)
        print(result)
    
    elif args.mode == "phonemize":
        if not args.voice:
            parser.error("Phonemize mode requires --voice")
        result = phonemize_files(args.voice)
        print(result)
    
    elif args.mode == "archive":
        if not args.voice:
            parser.error("Archive mode requires --voice")
        archive_dataset(args.voice)
        print("Dataset archived successfully")
    
    elif args.mode == "update_config":
        if not all([args.voice, args.save_freq, args.log_interval, args.epochs, args.batch_size]):
            parser.error("Update config mode requires --voice")
        result = update_config(args.voice, args.save_freq, args.log_interval, args.epochs, args.batch_size, args.max_len, args.pretrained_model, args.load_only_params, args.F0_path, args.ASR_config, args.ASR_path, args.PLBERT_dir, args.train_data, args.val_data, args.root_path, args.diff_epoch, args.joint_epoch)
        print(result)
    
    elif args.mode == "train":
        if not args.voice:
            parser.error("Train mode requires --voice")
        result = start_training(args.voice)
        print(result)

if __name__ == "__main__":
    main()
