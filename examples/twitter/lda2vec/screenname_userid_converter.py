"""
This Module is for finding userid by using screen_name/username.

+------------+
|Dependencies|
+------------+

	* Twython (pip install twython
	* OAuth (usually gets installed with Twython)

.. moduleauthor:: Ankur Tyagi (warlock_ankur)
"""

from twython import Twython
import logging
import time
APP_KEY = "f1extHsSnutNDhQdvdYLTnUZP"
APP_SECRET = "r4jTRVjkao4wknlIDI6riYVScBuLbkRGycw6gTZbALCHeudN8Y"

READFILE = "mentioned_user_names.txt"
WRITEFILE = "mentioned_user_userids.txt"


def getScreenNames():
    import pickle

    with open(READFILE, "rb") as f:
        screen_names = pickle.load(f)

    return screen_names


# Have not Used Show_User because that gives us only 180req while this can give us total of 600req
def lookupUser(names):
    twitter = Twython(APP_KEY, APP_SECRET)
    users = []
    try:
        users = twitter.lookup_user(screen_name=names)
    except Exception as e:
        logging.error("Error Occured while fetching the Users. Error = %s", e)
        if "No user matches" in e.message:
            for name in names:
                try:
                    user = twitter.lookup_user(screen_name=name)
                    users.append(user)
                except Exception as e:
                    if "No user matches" in e.message:
                        continue
                        #skip this name
    users = None if len(users) == 0 else users
    return users


def userlimitexceed(names):
    users = []
    lst = 0
    i = 100
    while i<=len(names):
    #for i in range(100, len(names), 100):
        print("Retrieving Userids for users "+str(lst) + " till " + str(i))
        newnames = names[lst:i]
        _users = lookupUser(newnames)
        if _users == None:
            logging.info("Rate Limited. Going to Sleep for 15 min.")
            print "Rate Limited. Going to Sleep for 15 min."
            time.sleep(15 * 60)
            logging.info("Waked Up!!. Restarting the retrieval Process.")
            print "Waked Up!!. Restarting the retrieval Process."
            continue

        for user in _users:
            users.append(user)
        lst = i

        if i + 100 < len(names):
            i+= 100
        else:
            i = len(names) - i

    writeUserIds(users)


def writeUserIds(users):
    import pickle

    mapping = {}
    for user in users:
        mapping[user['screen_name']] = user['id_str']

    with open(WRITEFILE, "wb") as f:
        pickle.dump(mapping, f, pickle.HIGHEST_PROTOCOL)


def appendUserIds():
    names = getScreenNames()
    userlimitexceed(names)


if __name__ == '__main__':
    logging.basicConfig(filename="TwitterUserId.log", format='%(asctime)s %(message)s', level=logging.DEBUG)
    appendUserIds()