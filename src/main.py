import click
import subprocess
import os
from pathlib import Path

from functools import update_wrapper


def pass_obj(f):
    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        return ctx.invoke(f, ctx.obj, *args, **kwargs)
    return update_wrapper(new_func, f)


@click.command()
@click.option('--task', prompt='Task number, default is ',
              default=1,
              help='Task number, default is 1. Could be as well 2 for this one')
def mk_path(task):
    click.echo('Hello %s!' % task)
    if task == 1:
        path = "assignment_1"
    elif task == 2:
        path = "assignment_2"
    else:
        msg = f"task {task} doesn't exist, please try 1 or 2"
        raise click.BadParameter(msg)
    click.echo('path saved %s!' % path)
    click.echo('Path at terminal when executing this file')
    click.echo( Path(__file__).resolve().parent.parent )
    path_to_main = Path(__file__).resolve().parent.parent
    path_to_data = os.path.join(path_to_main, "data",
    	"raw", path)
    click.echo(path_to_data)

if __name__ == '__main__':
    mk_path()
