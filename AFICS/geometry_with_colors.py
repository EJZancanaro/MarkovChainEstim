#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure complète des géométries de coordination avec palette de couleurs.

Inclut : CN, nom, abréviation et couleur associée (HEX + RGB).

"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List


@dataclass
class Color:
    """Classe représentant une couleur avec différents formats (HEX, RGB, nom)."""
    hex_code: str
    rgb: Tuple[int, int, int]
    name: str

    def __str__(self) -> str:
        return f"{self.name} ({self.hex_code})"


@dataclass
class Geometry:
    """Classe représentant une géométrie de coordination."""
    cn: int
    name: str
    shortname: str
    color: Color


# Palette de couleurs par CN (gradient bleu → magenta)
CN_COLORS: Dict[int, Color] = {
    3: Color(hex_code="#1E88E5", rgb=(30, 136, 229), name="Bleu"),
    4: Color(hex_code="#00ACC1", rgb=(0, 172, 193), name="Cyan"),
    5: Color(hex_code="#43A047", rgb=(67, 160, 71), name="Vert"),
    6: Color(hex_code="#7CB342", rgb=(124, 179, 66), name="Vert clair"),
    7: Color(hex_code="#FDD835", rgb=(253, 216, 53), name="Jaune"),
    8: Color(hex_code="#FB8C00", rgb=(251, 140, 0), name="Orange"),
    9: Color(hex_code="#E53935", rgb=(229, 57, 53), name="Rouge"),
    10: Color(hex_code="#8E24AA", rgb=(142, 36, 170), name="Violet"),
    12: Color(hex_code="#D81B60", rgb=(216, 27, 96), name="Rose magenta"),
}

# Dictionnaire complet des géométries (nom → Geometry)
GEOMETRIES: Dict[str, Geometry] = {}


def _add_geom(cn: int, name: str, shortname: str) -> None:
    """Ajoute une géométrie dans le dictionnaire global."""
    GEOMETRIES[name.lower()] = Geometry(
        cn=cn,
        name=name,
        shortname=shortname,
        color=CN_COLORS[cn],
    )


# CN = 3
_add_geom(3, "trigonal-planar", "TP")

# CN = 4
_add_geom(4, "square-planar", "SP")
_add_geom(4, "tetrahedral", "T")

# CN = 5
_add_geom(5, "square-pyramidal", "SP")
_add_geom(5, "trigonal-bipyramidal", "TB")

# CN = 6
_add_geom(6, "octahedral", "O")
_add_geom(6, "trigonal-prismatic", "TP")

# CN = 7
_add_geom(7, "capped-octahedral", "CO")
_add_geom(7, "capped-trigonal-prismatic", "CTP")
_add_geom(7, "pentagonal-bipyramidal", "PB")

# CN = 8
_add_geom(8, "cubic", "C")
_add_geom(8, "square-antiprismatic", "SA")
_add_geom(8, "bicapped-trigonal-prismatic", "BTP")
_add_geom(8, "dodecahedral", "D")
_add_geom(8, "hexagonal-bipyramidal", "HB")
_add_geom(8, "trans-bicapped-octahedral", "TBO")

# CN = 9
_add_geom(9, "capped-square-antiprismatic", "CSA")
_add_geom(9, "capped-square", "CS")
_add_geom(9, "tricapped-trigonal-prismatic", "TTP")

# CN = 10
_add_geom(10, "bicapped-square-antiprismatic", "BSA")
_add_geom(10, "bicapped-square", "BS")
_add_geom(10, "octagonal-bipyramidal", "OB")
_add_geom(10, "pentagonal-antiprismatic", "PA")
_add_geom(10, "pentagonal-prismatic", "PP")

# CN = 12
_add_geom(12, "icosahedral", "I")


def get_geometry(name: str) -> Optional[Geometry]:
    """Retourne une géométrie par son nom (insensible à la casse)."""
    return GEOMETRIES.get(name.lower())


def get_cn(name: str) -> Optional[int]:
    """Retourne le CN d'une géométrie, ou None si inconnue."""
    geom = GEOMETRIES.get(name.lower())
    return geom.cn if geom else None


def get_color(name: str) -> Optional[Color]:
    """Retourne la couleur associée à une géométrie, ou None si inconnue."""
    geom = GEOMETRIES.get(name.lower())
    return geom.color if geom else None


def get_geometries_by_cn(cn: int) -> List[Geometry]:
    """Retourne toutes les géométries pour un CN donné."""
    return [g for g in GEOMETRIES.values() if g.cn == cn]


if __name__ == "__main__":
    # Petit affichage de contrôle / debug
    print("=" * 70)
    print("GÉOMÉTRIES DE COORDINATION AVEC COULEURS (CN 3-12)")
    print("=" * 70)

    for cn in sorted({g.cn for g in GEOMETRIES.values()}):
        color = CN_COLORS[cn]
        geoms = get_geometries_by_cn(cn)
        print()
        print(f"CN = {cn} | Couleur : {color.name:<15} | HEX : {color.hex_code}")
        print("-" * 60)
        for geom in geoms:
            print(f"  {geom.name:<35} : {geom.shortname}")

    print()
    print("=" * 70)
    print(f"Total : {len(GEOMETRIES)} géométries")

