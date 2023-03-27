#!/usr/bin/env python

from __future__ import annotations

import os

import gradio as gr

from constants import MODEL_LIBRARY_ORG_NAME, SAMPLE_MODEL_REPO, UploadTarget
from inference import InferencePipeline
from trainer import Trainer


def create_training_demo(trainer: Trainer,
                         pipe: InferencePipeline | None = None) -> gr.Blocks:
    hf_token = os.getenv('HF_TOKEN')
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown('Training Data')
                    training_video = gr.File(label='Training video')
                    training_prompt = gr.Textbox(
                        label='Training prompt',
                        max_lines=1,
                        placeholder='A man is skiing')
                    gr.Markdown('''
                        - Upload a video and write a `Training Prompt` that describes the video.
                        ''')

            with gr.Column():
                with gr.Box():
                    gr.Markdown('Training Parameters')
                    with gr.Row():
                        base_model = gr.Text(
                            label='Base Model',
                            value='CompVis/stable-diffusion-v1-4',
                            max_lines=1)
                        resolution = gr.Dropdown(choices=['512', '768'],
                                                 value='512',
                                                 label='Resolution',
                                                 visible=False)
                    with gr.Row():
                        tuned_model = gr.Text(
                            label='Path to tuned model',
                            value='xxx/ski-lego',
                            max_lines=1)
                        resolution = gr.Dropdown(choices=['512', '768'],
                                                 value='512',
                                                 label='Resolution',
                                                 visible=False)

                    input_token = gr.Text(label='Hugging Face Write Token',
                                          placeholder='',
                                          visible=False if hf_token else True)
                    with gr.Accordion('Advanced settings', open=False):
                        num_training_steps = gr.Number(
                            label='Number of Training Steps',
                            value=300,
                            precision=0)
                        learning_rate = gr.Number(label='Learning Rate',
                                                  value=0.000035)
                        cross_replace = gr.Number(label='Cross attention replace ratio',
                                                  value=0.2)
                        gradient_accumulation = gr.Number(
                            label='Number of Gradient Accumulation',
                            value=1,
                            precision=0)
                        seed = gr.Slider(label='Seed',
                                         minimum=0,
                                         maximum=100000,
                                         step=1,
                                         randomize=True,
                                         value=0)
                        fp16 = gr.Checkbox(label='FP16', value=True)
                        use_8bit_adam = gr.Checkbox(label='Use 8bit Adam',
                                                    value=False)
                        checkpointing_steps = gr.Number(
                            label='Checkpointing Steps',
                            value=1000,
                            precision=0)
                        validation_epochs = gr.Number(
                            label='Validation Epochs', value=300, precision=0)
                    gr.Markdown('''
                        - The base model must be a Stable Diffusion model compatible with [diffusers](https://github.com/huggingface/diffusers) library.
                        - Expected time to complete: ~20 minutes with T4.
                        - You can check the training status by pressing the "Open logs" button if you are running this on your Space.
                        - Find the official github code [here](https://github.com/ShaoTengLiu/Video-P2P).
                        ''')

        with gr.Row():
            with gr.Column():
                gr.Markdown('Output Model')
                output_model_name = gr.Text(label='Path to save your tuned model',
                                            placeholder='ski-lego',
                                            max_lines=1)
                validation_prompt = gr.Text(
                    label='Validation Prompt',
                    placeholder=
                    'prompt to test the model, e.g: a Lego man is surfing')
                blend_word_1 = gr.Text(
                    label='blend_word(source)',
                    placeholder=
                    'man')
                blend_word_2 = gr.Text(
                    label='blend_word(target)',
                    placeholder=
                    'man')
                eq_params_1 = gr.Text(
                    label='reweight_word',
                    placeholder=
                    'Lego')
                eq_params_2 = gr.Text(
                    label='reweight_value',
                    placeholder=
                    '4')
            with gr.Column():
                gr.Markdown('Upload Settings')
                with gr.Row():
                    upload_to_hub = gr.Checkbox(label='Upload model to Hub',
                                                value=True)
                    use_private_repo = gr.Checkbox(label='Private', value=True)
                    delete_existing_repo = gr.Checkbox(
                        label='Delete existing repo of the same name',
                        value=False)
                    upload_to = gr.Radio(
                        label='Upload to',
                        choices=[_.value for _ in UploadTarget],
                        value=UploadTarget.MODEL_LIBRARY.value)

        remove_gpu_after_training = gr.Checkbox(
            label='Remove GPU after training',
            value=False,
            interactive=bool(os.getenv('SPACE_ID')),
            visible=False)
        run_button = gr.Button('Start Tuning')

        with gr.Box():
            gr.Markdown('Output message')
            output_message = gr.Markdown()

        if pipe is not None:
            run_button.click(fn=pipe.clear)
        run_button.click(
            fn=trainer.run,
            inputs=[
                training_video, training_prompt, output_model_name,
                delete_existing_repo, validation_prompt, base_model,
                resolution, num_training_steps, learning_rate,
                gradient_accumulation, seed, fp16, use_8bit_adam,
                checkpointing_steps, validation_epochs, upload_to_hub,
                use_private_repo, delete_existing_repo, upload_to,
                remove_gpu_after_training, input_token, blend_word_1, blend_word_2, eq_params_1, eq_params_2
            ],
            outputs=output_message)
        
        run_button_p2p = gr.Button('Start P2P')
        run_button_p2p.click(
            fn=trainer.run_p2p,
            inputs=[
                training_video, training_prompt, output_model_name,
                delete_existing_repo, validation_prompt, base_model,
                resolution, num_training_steps, learning_rate,
                gradient_accumulation, seed, fp16, use_8bit_adam,
                checkpointing_steps, validation_epochs, upload_to_hub,
                use_private_repo, delete_existing_repo, upload_to,
                remove_gpu_after_training, input_token, blend_word_1, blend_word_2, eq_params_1, eq_params_2, tuned_model, cross_replace
            ],
            outputs=output_message)
    return demo


if __name__ == '__main__':
    hf_token = os.getenv('HF_TOKEN')
    trainer = Trainer(hf_token)
    demo = create_training_demo(trainer)
    demo.queue(max_size=1).launch(share=False)
