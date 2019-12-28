Motivation

This project aims to prove mathematical statements
using Deep Neural Networks (DNNs).

Recently, [DeepCubeA](https://www.nature.com/articles/s42256-019-0070-z.epdf?shared_access_token=-pCSsZa_J9bM8VyXLZLRctRgN0jAjWel9jnR3ZoTv0Osb8UCgUm5AQaSCMHWqWzsyV3KBcb13SAW-9IL1pAGd1HcSk40JSEjhoaBAi0ePvYh_5Dul6LvK0oJY1KI0ULo9O9HCut_y7aCTc93Th8m5g%3D%3D)
managed to find solutions to the Rubik's
cube using A-Star graph search algorithm,
where a DNN learns the heuristic via
value iteration.

Since the search space for the Rubik's
cube is large,
this opens the door for solving other
large graph searching problems.

It has also been shown that the [tranformer
model can learn to solve calculus equations](https://arxiv.org/pdf/1912.01412.pdf),
beating computer algebra systems
including Mathematica's.

Now, every formal proof can be expressed
as a graph problem,
where nodes are mathematical statements
that can be derived from a set of
axioms and rules.
Using this idea,
and combining the results of the previous papers,
we are trying to prove that an automated
prover is possible.
