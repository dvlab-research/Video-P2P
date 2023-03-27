#!/usr/bin/env python

from __future__ import annotations

import pathlib

import gradio as gr
import slugify

from constants import MODEL_LIBRARY_ORG_NAME, UploadTarget
from uploader import Uploader
from utils import find_exp_dirs


class ModelUploader(Uploader):
    def upload_model(
        self,
        folder_path: str,
        repo_name: str,
        upload_to: str,
        private: bool,
        delete_existing_repo: bool,
        input_token: str | None = None,
    ) -> str:
        if not folder_path:
            raise ValueError
        if not repo_name:
            repo_name = pathlib.Path(folder_path).name
        repo_name = slugify.slugify(repo_name)

        if upload_to == UploadTarget.PERSONAL_PROFILE.value:
            organization = ''
        elif upload_to == UploadTarget.MODEL_LIBRARY.value:
            organization = MODEL_LIBRARY_ORG_NAME
        else:
            raise ValueError

        return self.upload(folder_path,
                           repo_name,
                           organization=organization,
                           private=private,
                           delete_existing_repo=delete_existing_repo,
                           input_token=input_token)


def load_local_model_list() -> dict:
    choices = find_exp_dirs()
    return gr.update(choices=choices, value=choices[0] if choices else None)


def create_upload_demo(hf_token: str | None) -> gr.Blocks:
    uploader = ModelUploader(hf_token)
    model_dirs = find_exp_dirs()

    with gr.Blocks() as demo:
        with gr.Box():
            gr.Markdown('Local Models')
            reload_button = gr.Button('Reload Model List')
            model_dir = gr.Dropdown(
                label='Model names',
                choices=model_dirs,
                value=model_dirs[0] if model_dirs else None)
        with gr.Box():
            gr.Markdown('Upload Settings')
            with gr.Row():
                use_private_repo = gr.Checkbox(label='Private', value=True)
                delete_existing_repo = gr.Checkbox(
                    label='Delete existing repo of the same name', value=False)
            upload_to = gr.Radio(label='Upload to',
                                 choices=[_.value for _ in UploadTarget],
                                 value=UploadTarget.MODEL_LIBRARY.value)
            model_name = gr.Textbox(label='Model Name')
            input_token = gr.Text(label='Hugging Face Write Token',
                                  placeholder='',
                                  visible=False if hf_token else True)
        upload_button = gr.Button('Upload')
        gr.Markdown(f'''
            - You can upload your trained model to your personal profile (i.e. https://huggingface.co/{{your_username}}/{{model_name}}) or to the public [Video-P2P Library](https://huggingface.co/{MODEL_LIBRARY_ORG_NAME}) (i.e. https://huggingface.co/{MODEL_LIBRARY_ORG_NAME}/{{model_name}}).
            ''')
        with gr.Box():
            gr.Markdown('Output message')
            output_message = gr.Markdown()

        reload_button.click(fn=load_local_model_list,
                            inputs=None,
                            outputs=model_dir)
        upload_button.click(fn=uploader.upload_model,
                            inputs=[
                                model_dir,
                                model_name,
                                upload_to,
                                use_private_repo,
                                delete_existing_repo,
                                input_token,
                            ],
                            outputs=output_message)

    return demo


if __name__ == '__main__':
    import os

    hf_token = os.getenv('HF_TOKEN')
    demo = create_upload_demo(hf_token)
    demo.queue(max_size=1).launch(share=False)
