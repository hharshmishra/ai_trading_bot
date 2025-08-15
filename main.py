from agents.news_agent import get_news_signal
from agents.indicator_agent import get_indicator_signal
from agents.pattern_agent import get_pattern_signal
from brain.decision_maker import decide

def main():
    news_data = get_news_signal()
    indicator_data = get_indicator_signal()
    pattern_data = get_pattern_signal()

    result = decide(news_data, indicator_data, pattern_data)
    print(result)

if __name__ == "__main__":
    main()
