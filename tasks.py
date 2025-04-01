from invoke import task


@task
def req_compile(ctx):
    """
    Compile Python requirements without upgrading.
    """
    ctx.run("pip-compile requirements/requirements.in")


@task
def req_upgrade(ctx):
    """
    Compile Python requirements with upgrading.
    """
    ctx.run("pip-compile -U requirements/requirements.in")


@task
def build(ctx):
    """
    Install all dependencies.
    """
    ctx.run("pip install -r requirements/requirements.txt")


@task
def rebuild(ctx):
    """
    Compile and rebuild the environment dependencies.
    """
    ctx.run("inv req-compile && inv build")


@task
def lint(ctx, path="src"):
    """
    Lint project files.
    """
    ctx.run("pylint --generate-rcfile > .pylintrc", warn=True)
    ctx.run(f"pylint --fail-under=8.0 --rcfile=.pylintrc {path}")


@task(
    help={
        "path": "Specify files or directories to lint",
        "check": "Only runs check without reformat (default: False)",
    }
)
def lint_black(ctx, path="src", check=False):
    """
    Runs the Black Formatter.
    """

    cmd = "black --line-length=100 --skip-string-normalization {check} {path}".format(
        check="--check" if check else "", path=path
    )
    ctx.run(cmd)
