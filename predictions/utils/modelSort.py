def cmp(kid1, kid2):
    if kid1.score == kid2.score:
        return kid1.age < kid2.age
    else:
        return kid1.score > kid2.score