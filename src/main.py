import click
import subprocess
import os
import logging
import json
import pandas as pd
from subprocess import Popen, PIPE, STDOUT
from pathlib import Path
from functools import update_wrapper

""" Functions to create cmd for subprocess """


def mk_cmd_paste(path_to_raw, path_to_data, sample=None):
    if sample:
        cmd = " ".join([
            "paste -d ' '",
            " {}".format(os.path.join(path_to_raw, "y.txt")),
            " {}".format(os.path.join(path_to_raw, "X.txt")),
            "| mungy sample {}".format(sample),
            "-o {}".format(os.path.join(path_to_data, "yXsample.txt"))
        ])
    else:
        cmd = " ".join([
            "paste -d ' '",
            " {}".format(os.path.join(path_to_raw, "y.txt")),
            " {}".format(os.path.join(path_to_raw, "X.txt")),
            "> {}".format(os.path.join(path_to_data, "data_all.txt"))
        ])
    return cmd
""" For one time? at least code looks better """


def mk_mungy_split(path_to_data, files):
    cmd = " ".join([
        "mungy split",
        "--no-copy-header",
        "--no-skip-header",
        "-o {}".format(path_to_data),
        "--filenames {}".format("files"),
        " {} ".format(os.path.join(path_to_data, "yXsample.txt")),
        "0.6,0.2,0.2"
    ])
    return cmd

""" This is fine... """


def mk_vw_file(path_to_data, file):
    cmd = " ".join([
        "mungy csv2vw",
        "{}.txt".format(os.path.join(path_to_data, file)),
        "--delim=' '",
        "--label=0",
        "--no-binary-label",
        "--no-header",
        "-o {}.vw".format(os.path.join(path_to_data, file))
    ])
    return cmd

""" Function to create path name for task """


def mk_path_task(task):
    click.echo("")
    click.echo('Task %s started!' % task)
    if task == 1:
        path = "assignment_1"
    elif task == 2:
        path = "assignment_2"
    else:
        click.echo("")
        click.echo("Something went wrong:")
        msg = f"task {task} doesn't exist, please try 1 or 2"
        raise click.BadParameter(msg)
    path_to_main = Path(__file__).resolve().parent.parent
    path_to_raw_data = os.path.join(path_to_main, "data",
                                    "raw", path)
    path_to_processed_data = os.path.join(path_to_main, "data",
                                          "processed", path)
    return path_to_raw_data, path_to_processed_data

""" Function to check if we have for task raw data """


def chk_raw_data(path_to_data):
    click.echo('path to raw data: %s' % path_to_data)
    if os.path.isdir(path_to_data):
        x_exist = os.path.isfile(os.path.join(path_to_data, "X.txt"))
        y_exist = os.path.isfile(os.path.join(path_to_data, "y.txt"))
        if x_exist and y_exist:
            click.echo('')
            click.echo('All raw data files for assignment task exist')
        else:
            click.echo("")
            click.echo("Something went wrong:")
            msg = f"Raw data files doesn't exist in {path_to_data}, please add them"
            raise click.BadParameter(msg)
    else:
        os.mkdir(path_to_data)
        click.echo("")
        click.echo("Something went wrong:")
        msg = f"created {path_to_data} folder, but please add X.txt and y.txt"
        raise click.BadParameter(msg)

""" checking for processed data, if everything is fine"""


def chk_processed_data(path_to_data, path_to_raw, noskip):
    click.echo('path to data: lalala')
    if (not os.path.isdir(path_to_data)):
        os.mkdir(path_to_data)
        click.echo('Path created')
        logger.info('We need to create all files, it will take a while')
    data_all_txt = os.path.isfile(os.path.join(path_to_data, "data_all.txt"))
    data_all_vw = os.path.isfile(os.path.join(path_to_data, "data_all.vw"))
    yXsample_txt = os.path.isfile(os.path.join(path_to_data, "yXsample.txt"))
    training_txt = os.path.isfile(os.path.join(path_to_data, "training.txt"))
    training_vw = os.path.isfile(os.path.join(path_to_data, "training.vw"))
    testing_txt = os.path.isfile(os.path.join(path_to_data, "testing.txt"))
    testing_vw = os.path.isfile(os.path.join(path_to_data, "testing.vw"))
    validation_txt = os.path.isfile(
        os.path.join(path_to_data, "validation.txt"))
    validation_vw = os.path.isfile(os.path.join(path_to_data, "validation.vw"))
    if data_all_txt and data_all_vw and training_txt and training_vw and \
            testing_txt and validation_vw and validation_txt and yXsample_txt:
        click.echo("All processed files are ready")
    else:
        click.echo("Something is missing")
        logger = logging.getLogger("creating")
        if noskip != "True":
            click.echo("Check for any lying cake-sniffers")
            chk_raw_data(path_to_raw)
            click.echo("No cake-sniffers detected")
        if (not yXsample_txt):
            logger.info("creating yXsample.txt file")
            subprocess.run(mk_cmd_paste(
                path_to_raw, path_to_data, 0.1), shell=True)
        if (not data_all_txt):
            logger.info("creating data_all.txt")
            subprocess.run(mk_cmd_paste(path_to_raw, path_to_data), shell=True)
        if (not validation_txt) or (not testing_txt) or (not training_txt):
            logger.info("creating all sets of txt")
            subprocess.run(mk_mungy_split(
                path_to_data, "training.txt,validation.txt,testing.txt"), shell=True)
        if (not validation_vw) or (not testing_vw) or (not training_vw):
            logger.info("creating all sets of training/testing vw")
            subprocess.run(mk_vw_file(path_to_data, "training"), shell=True)
            subprocess.run(mk_vw_file(path_to_data, "testing"), shell=True)
            subprocess.run(mk_vw_file(path_to_data, "validation"), shell=True)
        if (not data_all_vw):
            logger.info("creating data_all_vw")
            subprocess.run(mk_vw_file(path_to_data, "data_all"), shell=True)
        logger.info("all data which was required created")
        click.echo("We can go further")
# this could already get hidden into module..
""" some vw functions """


def mk_cmd_vw_train(path_model, path_readable_model, path_data=None, L1=None, L2=None, passes=20):
    cmd = " ".join([
        "vw",
        "--kill_cache",
        "--normalized",
        "--cache",
        "--holdout_off",
        "--quadratic nn",
        "--loss_function squared",
        "--passes {}".format(passes),
        "--data {}".format(path_data) if path_data and path_data != "-" else "",
        "--final_regressor {}".format(path_model),
        "--readable_model {}".format(path_readable_model),
        "--l1 {}".format(L1) if L1 else "",
        "--l2 {}".format(L2) if L2 else "",
    ])
    return cmd


def mk_cmd_vw_predict(path_model, path_predictions, path_data=None):
    cmd = " ".join([
        "vw",
        "--kill_cache",
        "--normalized",
        "--testonly",
        "--data {}".format(path_data) if path_data and path_data != "-" else "",
        "--initial_regressor {}".format(path_model),
        "--predictions {}".format(path_predictions),
    ])
    return cmd


def mk_cmd_metrics(path_predictions, path_initial, path_data=None):
    cmd = " ".join([
        "paste",
        "-d ' '",
        "{}".format(path_predictions),
        "{}".format(path_initial),
        "| mpipe metrics-reg",
    ])
    return cmd


def mk_cmd_invert_hashes(task_name, path_file, path_model):
    init_reg = os.path.join(
        path_model, "vw_serialised_models", "{}.vw_model".format(task_name))
    inv_has = os.path.join(path_model, "vw_inverted_hashes",
                           "{}.inver_hash".format(task_name))
    cmd = " ".join([
        "vw",
        "{}".format(path_file),
        "--initial_regressor {}".format(init_reg),
        "--testonly",
        "--normalized",
        "--invert_hash {}".format(inv_has)
    ])
    return cmd


def parse_statistics(std_output):
    tmp = str(std_output)
    stats = tmp.replace("'", "").replace("b", "").replace(
        "MSE,MAE", "").replace("\\n", "").split(",")
    mse = float(stats[0])
    mae = float(stats[1])
    return (mse, mae)


def mk_vw_train_validate_paths(path_to_models, path_to_processed_data, name):
    vw_model = "{}.vw_model".format(os.path.join(
        path_to_models, "vw_serialised_models", name))
    vw_readable_model = "{}.vw_readable_model".format(
        os.path.join(path_to_models, "vw_serialised_models", name))
    vw_predictions = "{}.txt".format(os.path.join(
        path_to_processed_data, "vw_predictions", name))
    return vw_model, vw_readable_model, vw_predictions


def magic_training(task):
    # should it be global?
    path_to_main = Path(__file__).resolve().parent.parent
    if task == 1:
        path = "assignment_1"
    elif task == 2:
        path = "assignment_2"
    path_to_processed_data = os.path.join(path_to_main, "data",
                                          "processed", path)
    setting_path = "{}.json".format(path)
    path_to_settings = os.path.join(path_to_main, "models",
                                    setting_path)
    path_to_models = os.path.join(path_to_main, "models",
                                  path)
    path_training_vw = os.path.join(path_to_processed_data, "training.vw")
    path_testing_vw = os.path.join(path_to_processed_data, "testing.vw")
    path_testing_txt = os.path.join(path_to_processed_data, "testing.txt")
    path_all_data_vw = os.path.join(path_to_processed_data, "data_all.vw")
    path_all_data_txt = os.path.join(path_to_processed_data, "data_all.txt")
    res_df = pd.DataFrame(columns=['task', 'mse', 'mae'])

    if os.path.isfile(path_to_settings):
        click.echo("Loading settings")
        with open(path_to_settings) as json_file:
            parameters = json.load(json_file)
            for hyperparams in parameters["hyperparams"]:
                task_name = hyperparams["name"]
                vw_model, vw_readable_model, vw_predictions = mk_vw_train_validate_paths(
                    path_to_models, path_to_processed_data, task_name)
                cmd_train_str = mk_cmd_vw_train(vw_model, vw_readable_model, path_training_vw,
                                                hyperparams["L1"], hyperparams["L2"], hyperparams["passes"])
                cmd_predict_str = mk_cmd_vw_predict(
                    vw_model, vw_predictions, path_testing_vw)
                cmd_metrics_str = mk_cmd_metrics(
                    vw_predictions, path_testing_txt)

                subprocess.run(cmd_train_str, shell=True)
                subprocess.run(cmd_predict_str, shell=True)
                p = Popen(cmd_metrics_str, shell=True, stdin=PIPE,
                          stdout=PIPE, stderr=STDOUT, close_fds=True)
                out = p.stdout.read()
                click.echo(out)
                stats = parse_statistics(out)
                mse = stats[0]
                mae = stats[1]
                res_df = res_df.append(
                    {'task': task_name, 'mse': mse, 'mae': mae}, ignore_index=True)
                click.echo("Task named: %s complete" % task_name)
            res_df = res_df.sort_values(by='mse', ascending=True)
            res_df.to_csv("{}best_models.csv".format(path_to_models))

            best_task = res_df['task'].iloc[0]
            cmd_invert_hashes_str = mk_cmd_invert_hashes(best_task, path_testing_vw, path_to_models)
            subprocess.run(cmd_invert_hashes_str, shell=True)

            click.echo("Need to try with all data model")
            best_df = pd.DataFrame(columns=['task', 'mse', 'mae'])
            vw_model, vw_readable_model, vw_predictions = mk_vw_train_validate_paths(
                path_to_models, path_to_processed_data, best_task)
            cmd_predict_all_str = mk_cmd_vw_predict(
                vw_model, vw_predictions, path_all_data_vw)
            cmd_metrics_all_str = mk_cmd_metrics(
                vw_predictions, path_all_data_txt)
            subprocess.run(cmd_predict_all_str, shell=True)
            p = Popen(cmd_metrics_all_str, shell=True, stdin=PIPE,
                      stdout=PIPE, stderr=STDOUT, close_fds=True)
            out = p.stdout.read()
            stats = parse_statistics(out)
            mse = stats[0]
            mae = stats[1]
            best_df = best_df({'task': best_task, 'mse': mse, 'mae': mae}, ignore_index=True)
            best_df.to_csv("{}best_results.csv".format(path_to_models))
            click.echo("That's all")

    else:
        click.echo("")
        click.echo("Something went wrong:")
        msg = "No settings file, it has to json"
        raise click.BadParameter(msg)


@click.command()
@click.option('--task', prompt='Task number, default is ',
              default=1,
              help='Task number, default is 1. Could be as well 2 for this one')
@click.option('--noskip', prompt='Check for raw files(recommended)',
              default="True",
              help='Check if raw files exist, recommended \
              to skip if he have all processed files and \
              deleted raw files to get free space')
def main(task, noskip):
    logger = logging.getLogger(__name__)
    logger.info('Parameters read, processing')
    path_raw_data, path_proc_data = mk_path_task(task)
    if noskip == 'True':
        chk_raw_data(path_raw_data)
    click.echo(
        "Now we will check for the processed data and create what is missing")
    click.echo("")
    chk_processed_data(path_proc_data, path_raw_data, noskip)
    # this all has to go to make_datasets...
    # maybe next time I will try harder...
    logger.info("Checking for config file and if ok - we go")
    magic_training(task)
    logger.info("End")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
