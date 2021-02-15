from bwsi_grader.python.card_games import grade_card

class Card:
    _rank_to_str = {11: 'Jack', 12: 'Queen', 13: 'King', 14: 'Ace'}
    _suit_to_str = {'C': 'Clubs', 'H': 'Hearts', 'S': 'Spades', 'D': 'Diamonds'}
    rank = 0;
    suit = "";

    def __init__(self, rank: int, suit: str):
        self.rank = rank;
        self.suit = suit.upper();

    def __repr__(self):
        ret = "";
        if (self.rank in self._rank_to_str): ret += self._rank_to_str[self.rank];
        else: ret += str(self.rank);
        ret += " of ";
        ret += self._suit_to_str[self.suit];
        return ret;

    def __lt__(self, other):
        return self.rank < other.rank;

    def __gt__(self, other):
        return self.rank > other.rank;

    def __le__(self, other):
        return self.rank <= other.rank;

    def __ge__(self, other):
        return self.rank >= other.rank;

    def __eq__(self, other):
        return self.rank == other.rank;

grade_card(Card);
