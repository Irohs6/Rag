import ast
from pathlib import Path


def doubler(*, nombre: int) -> int:
    return nombre * 2


def est_pair(nombre: int) -> bool:
    return nombre % 2 == 0


def somme_pairs(nombres: list[int]) -> int:
    total = 0
    for nombre in nombres:
        if est_pair(nombre):
            total += nombre
    return total


if __name__ == "__main__":
    # __file__ contient le chemin de ce fichier.
    text = Path(__file__).read_text(encoding="utf-8")
    arbre = ast.parse(text)
    for element in arbre.body:
        if isinstance(element, ast.FunctionDef):
            print(f"{element.name, element.args.kwonlyargs} : lignes {element.lineno} à {element.end_lineno}")
            print(ast.get_source_segment(text, element))
            print()