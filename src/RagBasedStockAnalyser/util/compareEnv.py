import yaml
from subprocess import run, PIPE

# Load environment.yml
with open("environment.yml") as f:
    env = yaml.safe_load(f)

declared = set()
pip_deps =set()
for dep in env["dependencies"]:
    if isinstance(dep, str):
        declared.add(dep.split("=")[0].lower())
    elif isinstance(dep, dict) and "pip" in dep:
        for pip_entry in dep["pip"]:
            if pip_entry.startswith("-r "):
                pip_file = pip_entry.split("-r ")[1]
                with open(pip_file) as pf:
                    pip_deps.update(line.strip() for line in pf if line.strip() and not line.startswith("#"))
            else:
                pip_deps.add(pip_entry)

# Get current environment
result = run(["conda", "list"], stdout=PIPE, text=True,shell=True)
installed = set(line.split()[0].lower() for line in result.stdout.splitlines() if not line.startswith("#"))

# Compare
missing = declared - installed
extra = installed - declared

print("Missing from env:", missing)
print("Extra in current:", extra)
