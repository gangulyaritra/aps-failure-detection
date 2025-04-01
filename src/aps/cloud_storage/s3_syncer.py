import os


class S3Sync:
    """
    Sync local folders with AWS S3 buckets using AWS CLI.
    """

    def sync_folder_to_s3(self, folder, aws_bucket_url):
        """
        Synchronizes a local folder to an AWS S3 bucket.

        :param folder: The local directory path to be synced.
        :param aws_bucket_url: The AWS S3 bucket URL.
        """
        command = f"aws s3 sync {folder} {aws_bucket_url}"
        os.system(command)

    def sync_folder_from_s3(self, folder, aws_bucket_url):
        """
        Synchronizes an AWS S3 bucket to a local folder.

        :param folder: The local directory path where data will be downloaded.
        :param aws_bucket_url: The AWS S3 bucket URL.
        """
        command = f"aws s3 sync {aws_bucket_url} {folder}"
        os.system(command)
