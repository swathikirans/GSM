import requests
import os

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = {'id' : id}, stream = True)
    token = get_confirm_token(response)

    if token:
        params = {'id' : id, 'confirm' : token}
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


if not os.path.exists('models'):
    os.makedirs('models')

download_file_from_google_drive('1rPma02uIiNI-TmL50rOyDDL1JOZeYPzT', 'models/something-v1_RGB_BNInception_avg_segment8_checkpoint.pth.tar')
download_file_from_google_drive('1anqY7A1PuUq_qleesy8OtTTkm_UoK23r', 'models/something-v1_RGB_BNInception_avg_segment16_checkpoint.pth.tar')
download_file_from_google_drive('14OplsnndeyUx7Uf4Hy82enPrhodEYXgo', 'models/something-v1_RGB_InceptionV3_avg_segment8_checkpoint.pth.tar')
download_file_from_google_drive('1t_NAC4WNEqA2XMm7bK3TUVByNLgnM5wJ', 'models/something-v1_RGB_InceptionV3_avg_segment12_checkpoint.pth.tar')
download_file_from_google_drive('1JCHJZK7r15yHCUiHKh3XJrrvibEGZhu4', 'models/something-v1_RGB_InceptionV3_avg_segment16_checkpoint.pth.tar')
download_file_from_google_drive('14uqOw9yAAuk9HDIsm-2NI7zEKKf4gstF', 'models/something-v1_RGB_InceptionV3_avg_segment24_checkpoint.pth.tar')
download_file_from_google_drive('1gbtufj34TA-2Pwxn7RGPWwkvtyK98Zx5', 'models/diving48_RGB_InceptionV3_avg_segment16_checkpoint.pth.tar')


# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) is not 3:
#         print("Usage: python google_drive.py drive_file_id destination_file_path")
#     else:
#         # TAKE ID FROM SHAREABLE LINK
#         file_id = sys.argv[1]
#         # DESTINATION FILE ON YOUR DISK
#         destination = sys.argv[2]
#         download_file_from_google_drive(file_id, destination)