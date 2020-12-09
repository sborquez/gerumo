from pathlib import Path
import subprocess
import shutil
from tqdm import tqdm

zip_cmd = 'zip -r {folder}.zip {folder}'

def compress_evaluation_folder(folder, skip_predictions=False, skip_recursive=False, verbose=True):
    evaluation_folder = Path(folder)
    if not evaluation_folder.exists(): raise ValueError("Folder doesn't exists")
    predictions_folder = evaluation_folder / "predictions"
    if not skip_predictions and predictions_folder:
        if verbose: print(f"compressing: {predictions_folder}")
        zip_cmd_formated = zip_cmd.format(folder=predictions_folder.name)
        process = subprocess.Popen(zip_cmd_formated.split(), stdout=subprocess.PIPE, cwd=evaluation_folder)
        output, error = process.communicate()
        if verbose: print(f"deleting: {predictions_folder}")
        shutil.rmtree(predictions_folder)
        if not skip_recursive:
            for telescope in ("LST_LSTCam", "MST_FlashCam", "SST1M_DigiCam"):
                telescope_folder = evaluation_folder / "telescopes" / telescope
                if not telescope_folder.exists(): continue
                telescope_predictions_folder = telescope_folder /"predictions"
                if verbose: print(f"compressing: {telescope_predictions_folder}")
                zip_cmd_formated = zip_cmd.format(folder=telescope_predictions_folder.name)
                process = subprocess.Popen(zip_cmd_formated.split(), stdout=subprocess.PIPE, cwd=telescope_folder)
                output, error = process.communicate()
                if verbose: print(f"deleting: {telescope_predictions_folder}")
                shutil.rmtree(telescope_predictions_folder)
    if verbose: print(f"compressing: {evaluation_folder}")
    zip_cmd_formated = zip_cmd.format(folder=evaluation_folder.name)
    process = subprocess.Popen(zip_cmd_formated.split(), stdout=subprocess.PIPE, cwd=evaluation_folder.parent)
    output, error = process.communicate()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compressn a evaluation folder.")
    ap.add_argument("-f", "--folder", required=True, help="Evaluation folder.")
    ap.add_argument("-P", "--skip_predictions", action='store_true', dest='skip_predictions')
    ap.add_argument("-R", "--skip_recursive", action='store_true', dest='skip_recursive')
    ap.add_argument("-v", "--verbose", action='store_true', dest="verbose")
    kwargs = vars(ap.parse_args()) 

    compress_evaluation_folder(**kwargs)