from googleads import adwords


def main(client):
    # Initialize appropriate service.
    customer_service = client.GetService('CustomerService', version='v201806')

    customer_list = customer_service.getCustomers()
    for customer_entry in customer_list:
        print("Id: {}, Name:{}, Conversion Tracking flag:{}".format(customer_entry['customerId'],
                                                                    customer_entry['descriptiveName'],
                                                                    customer_entry['conversionTrackingSettings']['usesCrossAccountConversionTracking']))


if __name__ == '__main__':
    # Initialize client object.
    adwords_client = adwords.AdWordsClient.LoadFromStorage()
    adwords_client.SetClientCustomerId('6541993810')
    main(adwords_client)
