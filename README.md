## 🖥 Installation (Windows 10/11)

### Step 1: Install Python

Download and install Python from:  
👉 https://www.python.org/downloads/windows/

✅ During install, **check** "Add Python to PATH"

---

### Step 2: Install Visual Studio Code

Download and install VS Code from:  
👉 https://code.visualstudio.com/

Install the following VS Code extensions:
- Python
- Pylance
---

### Step 3: Clone or Download This Project

```bash
cd %USERPROFILE%\Documents
git clone https://github.com/0xphantomotr/optimizer
cd ems-optimizer

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python optimizer.py