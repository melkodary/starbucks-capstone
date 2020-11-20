import pandas as pd
import datetime
import numpy as np
import math
import json
import seaborn as sns
import matplotlib.pyplot as plt
from progressbar import ProgressBar


def id_mapper(df):
    coded_dict = dict()
    cter = 1
    id_encoded = []

    for val in df['id']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter += 1

        id_encoded.append(coded_dict[val])
    return id_encoded


def preprocess(df):
    """
    This function should have common preprocessing on dataframe to be used by other preprocessing functions.
    - Only takes copy of df

    INPUT:
    df - (pandas df).

    OUTPUT:
    copy_df -  (pandas df) a copy of the original dataframe.
    """
    # use int ids instead of hashes.
    #     id_encoded = id_mapper(df)
    #     del df['id']
    #     df['id'] = id_encoded
    #     df.set_index('id', inplace=True)

    return df.copy()


def preprocess_portfolio(portfolio):
    """
    This function cleans the portfolio datafram.
    - Get dummies for offer_type and channels.

    INPUT:
    portfolio - (pandas df) this is a portfolio df.

    OUTPUT:
    df -  (pandas df) dataframe having the profile df cleaned.
    """
    df = preprocess(portfolio)

    # normalize portfolio duration to match that of transcript time.
    df['duration'] = df['duration'] * 24

    # get dummies for offer type and channels
    offer_type = pd.get_dummies(df['offer_type'])

    # https://stackoverflow.com/questions/29034928/pandas-convert-a-column-of-list-to-dummies
    channels = pd.get_dummies(df['channels'].apply(pd.Series).stack()).sum(level=0)

    # drop the old columns and add the new dummy columns
    df.drop(columns=['channels', 'offer_type'], inplace=True)
    df = pd.concat([channels, df, offer_type], axis=1)

    return df


def preprocess_profile(profile):
    """
    This function cleans the profile datafram.
    - It remove the noise data
    - Get dummies for gender
    - Parses became_member_on and splits onto several columns
    - Generates a new column called member_duration that represents for how long this user has been a member in days.

    INPUT:
    profile - (pandas df) this is a profile df.

    OUTPUT:
    df -  (pandas df) dataframe having the profile df cleaned.
    """

    df = preprocess(profile)

    # lets drop the noise entries such as 118 ages.
    # todo is there a way of filling in these data accurately without dropping the null rows?
    noise_data_idx = df.query('age == 118').index
    df.drop(index=noise_data_idx, inplace=True)

    # split gender into dummies. Since no missing values exist after removing 118, this should be safe.
    gender = pd.get_dummies(df['gender'], prefix='gender', prefix_sep='_')

    # Parse the date.
    dt = pd.to_datetime(df['became_member_on'], format='%Y%m%d')
    df['became_member_year'] = dt.dt.year
    df['became_member_month'] = dt.dt.month
    df['became_member_day'] = dt.dt.day

    # lets keep track of how long this user has been a member.
    df['member_duration'] = (datetime.datetime.today().date() - dt.dt.date).dt.days

    df.drop(columns=['gender', 'became_member_on'], inplace=True)

    # add the dummies to our profile.
    df = pd.concat([df, gender], axis=1)

    return df


def preprocess_transcript(transcript, profile_118):
    '''
    INPUT:
    transcript - (pandas df) this is a transcript df.

    OUTPUT:
    df -  (pandas df) dataframe having the transcript df cleaned.
    '''
    df = preprocess(transcript)

    df = df[~df['person'].isin(profile_118['id'].tolist())]
    events = pd.get_dummies(df.event)

    df['offer_id'] = df['value'].apply(lambda x: x.get('offer id') or x.get('offer_id'))
    df['amount'] = df['value'].apply(lambda x: x.get('amount'))
    df.drop(columns=['value'], inplace=True)

    df = pd.concat([df, events], axis=1)

    return df


def split_transcript_df(transcript):
    """
    This function splits transcript df into two dfs, one are the offers and the other is the transactions.

    INPUT:
    transcript - (pandas df) this is a transcript df.

    OUTPUT:
    offer_df -  (pandas df) dataframe having the transcation rows filtered out.
    transaction_df - (pandas df) dataframe having the transcation rows filtered out.
    """
    split_df = transcript.copy()

    # extract offers.
    offer_df = split_df[split_df['transaction'] != 1]
    offer_df = offer_df.drop(columns=['transaction', 'amount'])

    transaction_df = split_df[split_df['transaction'] == 1]
    transaction_df = transaction_df.drop(columns=['offer_id', 'offer completed', 'offer received', 'offer viewed'])

    return offer_df, transaction_df


def combine_data(transcript, portfolio, profile):
    offer_df, transaction_df = split_transcript_df(transcript)

    combined_data = []
    val_users = offer_df['person'].unique().tolist()

    for i, user in enumerate(val_users):
        user_offers = offer_df[offer_df['person'] == user]

        user_transactions = transaction_df[transaction_df['person'] == user]
        user_received_offers = user_offers[user_offers['offer received'] == 1]
        user_viewed_offers = user_offers[user_offers['offer viewed'] == 1]
        user_completed_offers = user_offers[user_offers['offer completed'] == 1]

        rows = []
        for _, offer in user_received_offers.iterrows():

            offer_details = portfolio.query("id == @offer.offer_id")
            offer_received_time = offer.time

            # calculate deadline for offer.
            duration = offer_details['duration'].tolist()[0]
            offer_deadline = offer_received_time + duration

            # find the transactions during the offer.
            viewed_during_offer = np.logical_and(user_viewed_offers['time'] >= offer_received_time,
                                                 user_viewed_offers['time'] <= offer_deadline)

            user_transaction_during_offer = user_transactions[
                np.logical_and(user_transactions['time'] >= offer_received_time,
                               user_transactions['time'] <= offer_deadline)]

            # incase it is an informational offer
            offers_completed = viewed_during_offer
            user_impression = np.logical_and(offers_completed.sum() > 0,
                                             user_transaction_during_offer['amount'].sum() > 0)

            # incase it is not an informational offer
            if offer_details.informational.values[0] == 0:
                offers_completed = np.logical_and(user_completed_offers['time'] >= offer_received_time,
                                                  user_completed_offers['time'] <= offer_deadline)

                user_impression = np.logical_and(viewed_during_offer.sum() > 0, offers_completed.sum() > 0) and \
                                  (user_transaction_during_offer['amount'].sum() >=
                                   offer_details['difficulty'].tolist()[0])

            summary = {'person': user,
                       'offer_id': offer.offer_id,
                       'user_impression': int(user_impression),
                       'total_amount': user_transaction_during_offer['amount'].sum()}

            if user_impression:
                summary['user_impression'] = 1.0

            combined_data.append(summary)

    combined_data = pd.DataFrame(combined_data)
    combined_data = combined_data.merge(profile, left_on='person', right_on='id', how='left')
    combined_data.drop(columns='id', inplace=True)
    combined_data = combined_data.merge(portfolio, left_on='offer_id', right_on='id', how='left')
    combined_data.drop(columns='id', inplace=True)

    return combined_data


def create_user_item_matrix(offer_df, transaction_df, portfolio):
    """
    This function creates user_by_offer matrix.

    INPUT:
    offer_df - (pandas df) dataframe to be translated into user-offer matrix.
    transaction_df - (pandas df)
    portfolio - (pandas df)
    profile - (pandas df)

    OUTPUT:
    user_by_offer -  (pandas df) dataframe having user vs offer matrix.
    """

    # create an empty matrix.
    user_by_offer = offer_df.groupby(['person', 'offer_id'])['event'].agg(lambda x: np.nan).unstack()

    # init progress bar because this will take too long.
    pbar = ProgressBar()

    for person in pbar(user_by_offer.index):
        user_offers = offer_df[offer_df['person'] == person]
        user_transactions = transaction_df[transaction_df['person'] == person]

        user_offer_received = user_offers[user_offers['offer received'] == 1]
        user_offer_viewed = user_offers[user_offers['offer viewed'] == 1]
        user_offer_completed = user_offers[user_offers['offer completed'] == 1]

        for index, offer in user_offer_received.iterrows():
            offer_id = offer.offer_id

            # 0 means that this user received this offer
            user_by_offer.loc[person, offer_id] = 0

            # find offer in portfolio.
            offer_details = portfolio.query("id == @offer_id")

            # calculate deadline for offer.
            duration = offer_details['duration'].tolist()[0]
            offer_received_time = offer.time
            offer_deadline = offer_received_time + duration

            # find the transactions during the offer.
            viewed_during_offer = np.logical_and(user_offer_viewed['time'] >= offer_received_time,
                                                 user_offer_viewed['time'] <= offer_deadline)

            user_transaction_during_offer = user_transactions[
                np.logical_and(user_transactions['time'] >= offer_received_time,
                               user_transactions['time'] <= offer_deadline)]

            user_impression = np.logical_and(viewed_during_offer.sum() > 0,
                                             user_transaction_during_offer['amount'].sum() > 0)

            if offer_details.informational.values[0] == 0:
                offers_completed = np.logical_and(user_offer_completed['time'] >= offer_received_time,
                                                  user_offer_completed['time'] <= offer_deadline)

                user_impression = np.logical_and(viewed_during_offer.sum() > 0,
                                                 offers_completed.sum() > 0) and \
                                  (user_transaction_during_offer['amount'].sum() >= offer_details['difficulty'].values[
                                      0])

            if user_impression:
                user_by_offer.loc[person, offer_id] += 1

    return user_by_offer


def get_user_by_offer_matrix(offer_df, transaction_df, portfolio, profile, filename='user_by_offer.pkl'):
    """
    This function loads the matrix from storage. If it does not exist, it calls create_user_item_matrix function
    and saves the result on disk afterwards.

    INPUT:
    df - (pandas df) dataframe to be fed into create_user_item_matrix
    filename - (string) location of pickle file to be saved.

    OUTPUT:
    user_by_offer -  (pandas df) dataframe having user vs offer matrix.
    """
    # load user_by_offer if exists.
    try:
        user_by_offer = pd.read_pickle(filename)
    except FileNotFoundError:
        print('File not found. Generating user_by_offer_matrix')
        user_by_offer = create_user_item_matrix(offer_df, transaction_df, portfolio, profile)

        # save user_by_offer
        pd.to_pickle(user_by_offer, filename)

    return user_by_offer


def main():
    # read in the json files
    portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
    profile = pd.read_json('data/profile.json', orient='records', lines=True)
    transcript = pd.read_json('data/transcript.json', orient='records', lines=True)

    profile_118 = profile.query('age == 118')

    processed_portfolio = preprocess_portfolio(portfolio)
    processed_profile = preprocess_profile(profile)
    processed_transcript = preprocess_transcript(transcript, profile_118)
    offer_df, transaction_df = split_transcript_df(processed_transcript)
    # user_by_offer = get_user_by_offer_matrix(offer_df, transaction_df, processed_portfolio, processed_profile,
    #                                          filename='user_by_offer_1.pkl')
    combine_data(processed_transcript, processed_portfolio, processed_profile)


if __name__ == '__main__':
    main()
