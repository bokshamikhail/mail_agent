from app.config import RUN_MODE
from app.pipeline import run_once, run_forever

if __name__ == '__main__':
    if RUN_MODE == 'forever':
        run_forever()
    else:
        print(run_once())
