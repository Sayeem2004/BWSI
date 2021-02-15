import random;
from bwsi_grader.python.card_games import grade_deck;

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

class Deck:
    cards = [];
    shuffled = False;
    dealt = 0;

    def __init__(self, shuffled: bool = False):
        for i in range(2,15): self.cards.append(Card(i,"D"));
        for i in range(2,15): self.cards.append(Card(i,"C"));
        for i in range(2,15): self.cards.append(Card(i,"H"));
        for i in range(2,15): self.cards.append(Card(i,"S"));
        self.shuffled = shuffled;
        if (shuffled): self.shuffle();
        dealt = 0;

    def shuffle(self):
        self.cards = random.sample(self.cards, k=52);

    def deal_card(self):
        if (self.dealt < 52):
            self.dealt += 1;
            return self.cards[self.dealt-1];
        else: return None;

    def __repr__(self):
        return "Deck(dealt " + str(self.dealt) + ", shuffled=" + str(self.shuffled) + ")";

    def reset(self):
        self.dealt = 0;
        self.shuffled = False;
        self.cards = [];
        for i in range(2,15): self.cards.append(Card(i,"D"))
        for i in range(2,15): self.cards.append(Card(i,"C"))
        for i in range(2,15): self.cards.append(Card(i,"H"))
        for i in range(2,15): self.cards.append(Card(i,"S"))


grade_deck(Deck);
def play_high_low_game():
    d = Deck(shuffled=True);
    p1 = d.deal_card();
    p2 = d.deal_card();
    print("It's a tie!" if p1 == p2 else f'Player {1 if p1 > p2 else 2} wins!');
    print(f'Player 1 had the {p1} and Player 2 had the {p2}');
play_high_low_game();
