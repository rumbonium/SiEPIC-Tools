echo "Component Creation Wizard"
echo "-------------------------"
echo "Welcome!"
echo "Type the new component's name, followed by [ENTER]: "

read module

mkdir $module
cd $module

touch __init__.py
echo "from .model import Model" >> __init__.py

touch model.py
contents="

class Model:
    def __init__(self):
        pass

    # TODO: Provide the arguments required by the model and implement get_s_params
    @staticmethod
    def get_s_params(*args, **kwargs):
        pass

    @staticmethod
    def about():
        message = \"About $module:\"
        print(message)
"
echo "$contents" >> model.py

echo "Please add '$module' to INSTALLED_COMPONENTS in 'components.py' 
(also include it in whichever component model it belongs to)."