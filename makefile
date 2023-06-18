DATE = $(shell date -I date -v-1m)

default: map

map:
	python get_stories_wayback.py -start $(DATE)
	python get_topics.py -start $(DATE) -name US_MAINSTREAM
	python topic2map.py -start $(DATE) -name US_MAINSTREAM