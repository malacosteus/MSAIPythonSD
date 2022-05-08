from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict


class Organisation:
    def __init__(self, professions={"CEO": 1}, vacancies={}) -> None:

        self.prf: defaultdict[str, int] = defaultdict(int, professions)
        self.vac: defaultdict[str, int] = defaultdict(int, vacancies)
        self.sz = sum(self.prf.values())

    def add(self, prf_add: dict[str, int]) -> None:
        """
        Add employees to corporation
        Input data:
        -----------
            prd_add - number of workers to add to each profession
        Example:
        -------
            corp = Organisation({"driver: 1"})
            corp
            >> Number of drivers is equal to 1
            corp.add({"manager": 100, "programmer": 2"})
            >> Number of drivers is equal to 1
            >> Number of managers is equal to 100
            >> Number of programmers is equal to 2
        """
        for name, val in prf_add.items():
            self.prf[name] += val
            self.sz += val

    def add_vacancie(self, vac_add: dict[str, int]) -> None:
        """
        Add vacancie for corporation
        Input data:
        -----------
            vac_add - number of needed workers to add into vacancies list
        """
        for name, val in vac_add.items():
            self.vac[name] += val


class Organisations_Modifier(ABC):
    pass


class dyn_Organisations_Modifier(Organisations_Modifier):
    @abstractmethod
    def run(self, Org1, Org2):
        pass


class static_Organisations_Modifier(Organisations_Modifier):
    @staticmethod
    @abstractmethod
    def run(Org1, Org2):
        pass


class Intersect_Org(static_Organisations_Modifier):
    @staticmethod
    def run(Org1: Organisation, Org2: Organisation) -> dict[str, int]:
        intrsct = Org1.vac.keys() & Org2.vac.keys()
        ret_dct: defaultdict[str, int] = defaultdict(int)
        for vac_name in intrsct:
            ret_dct[vac_name] += min(Org1.vac[vac_name], Org2.vac[vac_name])
        return ret_dct


class Union_Org(static_Organisations_Modifier):
    @staticmethod
    def run(Org1: Organisation, Org2: Organisation) -> dict[str, int]:
        ret_dct: defaultdict[str, int] = defaultdict(int)
        for vac_name in Org1.vac.keys():
            ret_dct[vac_name] += Org1.vac[vac_name]
        for vac_name in Org2.vac.keys():
            ret_dct[vac_name] += Org2.vac[vac_name]
        return ret_dct


class Difference_org(static_Organisations_Modifier):
    @staticmethod
    def run(Org1: Organisation, Org2: Organisation) -> dict[str, int]:
        diff = Org1.vac.keys() - Org2.vac.keys()
        ret_dct: defaultdict[str, int] = defaultdict(int)
        for vac_name in diff:
            ret_dct[vac_name] += Org1.vac[vac_name]
        return ret_dct


class more_Org(static_Organisations_Modifier):
    @staticmethod
    def run(Org1: Organisation, Org2: Organisation) -> bool:
        return Org1.sz > Org2.sz


class eq_Org(static_Organisations_Modifier):
    @staticmethod
    def run(Org1: Organisation, Org2: Organisation) -> bool:
        return Org1.sz == Org2.sz


class compare_corp(dyn_Organisations_Modifier):
    def __init__(
        self,
        class1: static_Organisations_Modifier = more_Org(),
        class2: static_Organisations_Modifier = eq_Org(),
        text1="Corp 1 have gretaer staff size than corp 2",
        text2="Corp 1 have same staff size as corp 2",
        text3="Corp 1 have smaller staff size than corp 2",
    ):
        self.cmp1 = class1
        self.cmp2 = class2
        self.text1 = text1
        self.text2 = text2
        self.text3 = text3

    def run(self, Org1: Organisation, Org2: Organisation) -> None:
        if self.cmp1.run(Org1, Org2):
            print(self.text1)
        elif self.cmp2.run(Org1, Org2):
            print(self.text2)
        else:
            print(self.text3)


def prnt_dct(dct: dict) -> None:
    if not dct:
        print("None")
        return
    for name, val in dct.items():
        print(f"{name}: {val}")


if __name__ == "__main__":
    corp_1 = Organisation({"manager": 1, "driver": 100, "accountant": 5})
    corp_2 = Organisation(
        {"manager": 1000, "programmer": 15, "accountant": 3, "cleaner": 9}
    )
    corp_1.add_vacancie({"Uber driver": 1000000000, "cleaner": 8, "accountant": 2})
    corp_2.add_vacancie(
        {
            "programmer ": 1000,
            "cleaner": 19,
            "accountant": 90,
            "cleaner": 6,
            "tester": 25,
        }
    )
    cmp = compare_corp()
    # 1. compare companies by staff size
    cmp.run(corp_1, corp_2)
    print("Let's increase staff size of Corp 2")
    corp_1.add({"driver": 1000000000})
    cmp.run(corp_1, corp_2)
    print("\n")

    # 2. Get companies vacancies list difference/union/intersection
    # Difference
    print("corp_1 - corp_2:")
    dct = Difference_org.run(corp_1, corp_2)
    prnt_dct(dct)
    print("\n")

    print("corp_2 - corp_1:")
    dct = Difference_org.run(corp_2, corp_1)
    prnt_dct(dct)
    print("\n")

    print("corp_2 - corp_2:")
    dct = Difference_org.run(corp_2, corp_2)
    prnt_dct(dct)
    print("\n")

    # Union
    print("Union of corp_1 and corp_2:")
    dct = Union_Org.run(corp_1, corp_2)
    prnt_dct(dct)
    print("\n")

    # Intersection
    print("Intersection of corp_1 and corp_2:")
    dct = Intersect_Org.run(corp_1, corp_2)
    prnt_dct(dct)
