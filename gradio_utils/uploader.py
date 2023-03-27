from __future__ import annotations

from huggingface_hub import HfApi


class Uploader:
    def __init__(self, hf_token: str | None):
        self.hf_token = hf_token

    def upload(self,
               folder_path: str,
               repo_name: str,
               organization: str = '',
               repo_type: str = 'model',
               private: bool = True,
               delete_existing_repo: bool = False,
               input_token: str | None = None) -> str:

        api = HfApi(token=self.hf_token if self.hf_token else input_token)

        if not folder_path:
            raise ValueError
        if not repo_name:
            raise ValueError
        if not organization:
            organization = api.whoami()['name']

        repo_id = f'{organization}/{repo_name}'
        if delete_existing_repo:
            try:
                api.delete_repo(repo_id, repo_type=repo_type)
            except Exception:
                pass
        try:
            api.create_repo(repo_id, repo_type=repo_type, private=private)
            api.upload_folder(repo_id=repo_id,
                              folder_path=folder_path,
                              path_in_repo='.',
                              repo_type=repo_type)
            url = f'https://huggingface.co/{repo_id}'
            message = f'Your model was successfully uploaded to <a href="{url}" target="_blank">{url}</a>.'
        except Exception as e:
            message = str(e)
        return message
