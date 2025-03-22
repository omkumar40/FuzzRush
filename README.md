# 🚀 FuzzRush - High-Speed Fuzzy Matching

FuzzRush is a blazing-fast fuzzy string matching library leveraging **TF-IDF** and **sparse matrix operations** for scalable similarity calculations.

## 🔥 Features
- **Optimized fuzzy matching** using `sparse_dot_topn`
- **Fast TF-IDF vectorization** for large datasets
- **Configurable n-gram tokenization**
- **Multiple output formats (DataFrame & Dictionary)**

## 🛠 Installation
```bash
git clone https://github.com/your-username/FuzzRush.git
cd FuzzRush
pip install -r requirements.txt
```

## 📜 Usage
```python
from FuzzRush.fuzzrush import FuzzRush
source = ["Apple Inc", "Microsoft Corp"]
target = ["Apple", "Microsoft", "Google"]
matcher = FuzzRush(source, target)
matcher.tokenize(n=3)
matches = matcher.match()
print(matches)
```

## 🤝 Contributing
Pull requests are welcome!

## 📝 License
MIT License © 2025 omkumar40