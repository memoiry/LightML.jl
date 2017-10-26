ENV["PYTHON"]=""
Pkg.add("Conda")
using Conda
Conda.add("python==2.7.13")
Conda.add("matplotlib")
Conda.add("scikit-learn")
Pkg.add("PyCall")
Pkg.build("PyCall")