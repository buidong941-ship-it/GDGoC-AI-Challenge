import competition.config
from competition.integrations.drive_upload import create_drive_token


if __name__ == "__main__":
    token_path = create_drive_token()
    print(f"Saved OAuth token to {token_path}")
