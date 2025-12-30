from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def connect_drive():
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("token.json")

    if gauth.credentials is None:
        # login lần đầu
        print("No credentials → Login Google...")
        gauth.CommandLineAuth()
    elif gauth.access_token_expired:
        print("Token expired → Refreshing...")
        gauth.Refresh()
    else:
        gauth.Authorize()

    gauth.SaveCredentialsFile("token.json")
    print("✔ Google Drive connected")

    return GoogleDrive(gauth)


def upload_test_file(drive):
    content = "Xin chào — đây là file test upload từ Python 😀"
    
    file = drive.CreateFile({'title': 'drive_test_upload.txt'})
    file.SetContentString(content)
    file.Upload()

    print("✔ File uploaded successfully")


if __name__ == "__main__":
    drive = connect_drive()
    upload_test_file(drive)
