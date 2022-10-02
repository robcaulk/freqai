# Exchange-specific Notes

This page combines common gotchas and Information which are exchange-specific and most likely don't apply to other exchanges.

## Exchange configuration

Freqtrade is based on [CCXT library](https://github.com/ccxt/ccxt) that supports over 100 cryptocurrency
exchange markets and trading APIs. The complete up-to-date list can be found in the
[CCXT repo homepage](https://github.com/ccxt/ccxt/tree/master/python).
However, the bot was tested by the development team with only a few exchanges.
A current list of these can be found in the "Home" section of this documentation.

Feel free to test other exchanges and submit your feedback or PR to improve the bot or confirm exchanges that work flawlessly..

Some exchanges require special configuration, which can be found below.

### Sample exchange configuration

A exchange configuration for "binance" would look as follows:

```json
"exchange": {
    "name": "binance",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "ccxt_config": {},
    "ccxt_async_config": {},
    // ... 
```

### Setting rate limits

Usually, rate limits set by CCXT are reliable and work well.
In case of problems related to rate-limits (usually DDOS Exceptions in your logs), it's easy to change rateLimit settings to other values.

```json
"exchange": {
    "name": "kraken",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "ccxt_config": {"enableRateLimit": true},
    "ccxt_async_config": {
        "enableRateLimit": true,
        "rateLimit": 3100
    },
```

This configuration enables kraken, as well as rate-limiting to avoid bans from the exchange.
`"rateLimit": 3100` defines a wait-event of 3.1s between each call. This can also be completely disabled by setting `"enableRateLimit"` to false.

!!! Note
    Optimal settings for rate-limiting depend on the exchange and the size of the whitelist, so an ideal parameter will vary on many other settings.
    We try to provide sensible defaults per exchange where possible, if you encounter bans please make sure that `"enableRateLimit"` is enabled and increase the `"rateLimit"` parameter step by step.

## Binance

Binance supports [time_in_force](configuration.md#understand-order_time_in_force).

!!! Tip "Stoploss on Exchange"
    Binance supports `stoploss_on_exchange` and uses `stop-loss-limit` orders. It provides great advantages, so we recommend to benefit from it by enabling stoploss on exchange.
    On futures, Binance supports both `stop-limit` as well as `stop-market` orders. You can use either `"limit"` or `"market"` in the `order_types.stoploss` configuration setting to decide which type to use.

### Binance Blacklist recommendation

For Binance, it is suggested to add `"BNB/<STAKE>"` to your blacklist to avoid issues, unless you are willing to maintain enough extra `BNB` on the account or unless you're willing to disable using `BNB` for fees.
Binance accounts may use `BNB` for fees, and if a trade happens to be on `BNB`, further trades may consume this position and make the initial BNB trade unsellable as the expected amount is not there anymore.

### Binance sites

Binance has been split into 2, and users must use the correct ccxt exchange ID for their exchange, otherwise API keys are not recognized.

* [binance.com](https://www.binance.com/) - International users. Use exchange id: `binance`.
* [binance.us](https://www.binance.us/) - US based users. Use exchange id: `binanceus`.

### Binance Futures

Binance has specific (unfortunately complex) [Futures Trading Quantitative Rules](https://www.binance.com/en/support/faq/4f462ebe6ff445d4a170be7d9e897272) which need to be followed, and which prohibit a too low stake-amount (among others) for too many orders.
Violating these rules will result in a trading restriction.

When trading on Binance Futures market, orderbook must be used because there is no price ticker data for futures.

``` jsonc
  "entry_pricing": {
      "use_order_book": true,
      "order_book_top": 1,
      "check_depth_of_market": {
          "enabled": false,
          "bids_to_ask_delta": 1
      }
  },
  "exit_pricing": {
      "use_order_book": true,
      "order_book_top": 1
  },
```

#### Binance futures settings

Users will also have to have the futures-setting "Position Mode" set to "One-way Mode", and "Asset Mode" set to "Single-Asset Mode".
These settings will be checked on startup, and freqtrade will show an error if this setting is wrong.

![Binance futures settings](assets/binance_futures_settings.png)

Freqtrade will not attempt to change these settings.

## Kraken

!!! Tip "Stoploss on Exchange"
    Kraken supports `stoploss_on_exchange` and can use both stop-loss-market and stop-loss-limit orders. It provides great advantages, so we recommend to benefit from it.
    You can use either `"limit"` or `"market"` in the `order_types.stoploss` configuration setting to decide which type to use.

### Historic Kraken data

The Kraken API does only provide 720 historic candles, which is sufficient for Freqtrade dry-run and live trade modes, but is a problem for backtesting.
To download data for the Kraken exchange, using `--dl-trades` is mandatory, otherwise the bot will download the same 720 candles over and over, and you'll not have enough backtest data.

Due to the heavy rate-limiting applied by Kraken, the following configuration section should be used to download data:

``` json
    "ccxt_async_config": {
        "enableRateLimit": true,
        "rateLimit": 3100
    },
```

!!! Warning "Downloading data from kraken"
    Downloading kraken data will require significantly more memory (RAM) than any other exchange, as the trades-data needs to be converted into candles on your machine.
    It will also take a long time, as freqtrade will need to download every single trade that happened on the exchange for the pair / timerange combination, therefore please be patient.

!!! Warning "rateLimit tuning"
    Please pay attention that rateLimit configuration entry holds delay in milliseconds between requests, NOT requests\sec rate.
    So, in order to mitigate Kraken API "Rate limit exceeded" exception, this configuration should be increased, NOT decreased.

## Bittrex

### Order types

Bittrex does not support market orders. If you have a message at the bot startup about this, you should change order type values set in your configuration and/or in the strategy from `"market"` to `"limit"`. See some more details on this [here in the FAQ](faq.md#im-getting-the-exchange-bittrex-does-not-support-market-orders-message-and-cannot-run-my-strategy).

Bittrex also does not support `VolumePairlist` due to limited / split API constellation at the moment.
Please use `StaticPairlist`. Other pairlists (other than `VolumePairlist`) should not be affected.

### Volume pairlist

Bittrex does not support the direct usage of VolumePairList. This can however be worked around by using the advanced mode with `lookback_days: 1` (or more), which will emulate 24h volume.

Read more in the [pairlist documentation](plugins.md#volumepairlist-advanced-mode).

### Restricted markets

Bittrex split its exchange into US and International versions.
The International version has more pairs available, however the API always returns all pairs, so there is currently no automated way to detect if you're affected by the restriction.

If you have restricted pairs in your whitelist, you'll get a warning message in the log on Freqtrade startup for each restricted pair.

The warning message will look similar to the following:

``` output
[...] Message: bittrex {"success":false,"message":"RESTRICTED_MARKET","result":null,"explanation":null}"
```

If you're an "International" customer on the Bittrex exchange, then this warning will probably not impact you.
If you're a US customer, the bot will fail to create orders for these pairs, and you should remove them from your whitelist.

You can get a list of restricted markets by using the following snippet:

``` python
import ccxt
ct = ccxt.bittrex()
lm = ct.load_markets()

res = [p for p, x in lm.items() if 'US' in x['info']['prohibitedIn']]
print(res)
```

## FTX

!!! Tip "Stoploss on Exchange"
    FTX supports `stoploss_on_exchange` and can use both stop-loss-market and stop-loss-limit orders. It provides great advantages, so we recommend to benefit from it.
    You can use either `"limit"` or `"market"` in the `order_types.stoploss` configuration setting to decide which type of stoploss shall be used.

### Using subaccounts

To use subaccounts with FTX, you need to edit the configuration and add the following:

``` json
"exchange": {
    "ccxt_config": {
        "headers": {
            "FTX-SUBACCOUNT": "name"
        }
    },
}
```

## Kucoin

Kucoin requires a passphrase for each api key, you will therefore need to add this key into the configuration so your exchange section looks as follows:

```json
"exchange": {
    "name": "kucoin",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "password": "your_exchange_api_key_password",
    // ...
}
```

Kucoin supports [time_in_force](configuration.md#understand-order_time_in_force).

!!! Tip "Stoploss on Exchange"
    Kucoin supports `stoploss_on_exchange` and can use both stop-loss-market and stop-loss-limit orders. It provides great advantages, so we recommend to benefit from it.
    You can use either `"limit"` or `"market"` in the `order_types.stoploss` configuration setting to decide which type of stoploss shall be used.

### Kucoin Blacklists

For Kucoin, it is suggested to add `"KCS/<STAKE>"` to your blacklist to avoid issues, unless you are willing to maintain enough extra `KCS` on the account or unless you're willing to disable using `KCS` for fees. 
Kucoin accounts may use `KCS` for fees, and if a trade happens to be on `KCS`, further trades may consume this position and make the initial `KCS` trade unsellable as the expected amount is not there anymore.

## Huobi

!!! Tip "Stoploss on Exchange"
    Huobi supports `stoploss_on_exchange` and uses `stop-limit` orders. It provides great advantages, so we recommend to benefit from it by enabling stoploss on exchange.

## OKX (former OKEX)

OKX requires a passphrase for each api key, you will therefore need to add this key into the configuration so your exchange section looks as follows:

```json
"exchange": {
    "name": "okx",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "password": "your_exchange_api_key_password",
    // ...
}
```

!!! Warning
    OKX only provides 100 candles per api call. Therefore, the strategy will only have a pretty low amount of data available in backtesting mode.

!!! Warning "Futures"
    OKX Futures has the concept of "position mode" - which can be Net or long/short (hedge mode).
    Freqtrade supports both modes (we recommend to use net mode) - but changing the mode mid-trading is not supported and will lead to exceptions and failures to place trades.
    OKX also only provides MARK candles for the past ~3 months. Backtesting futures prior to that date will therefore lead to slight deviations, as funding-fees cannot be calculated correctly without this data.

## Gate.io

!!! Tip "Stoploss on Exchange"
    Gate.io supports `stoploss_on_exchange` and uses `stop-loss-limit` orders. It provides great advantages, so we recommend to benefit from it by enabling stoploss on exchange..

Gate.io allows the use of `POINT` to pay for fees. As this is not a tradable currency (no regular market available), automatic fee calculations will fail (and default to a fee of 0).
The configuration parameter `exchange.unknown_fee_rate` can be used to specify the exchange rate between Point and the stake currency. Obviously, changing the stake-currency will also require changes to this value.

## All exchanges

Should you experience constant errors with Nonce (like `InvalidNonce`), it is best to regenerate the API keys. Resetting Nonce is difficult and it's usually easier to regenerate the API keys.

## Random notes for other exchanges

* The Ocean (exchange id: `theocean`) exchange uses Web3 functionality and requires `web3` python package to be installed:

```shell
$ pip3 install web3
```

### Getting latest price / Incomplete candles

Most exchanges return current incomplete candle via their OHLCV/klines API interface.
By default, Freqtrade assumes that incomplete candle is fetched from the exchange and removes the last candle assuming it's the incomplete candle.

Whether your exchange returns incomplete candles or not can be checked using [the helper script](developer.md#Incomplete-candles) from the Contributor documentation.

Due to the danger of repainting, Freqtrade does not allow you to use this incomplete candle.

However, if it is based on the need for the latest price for your strategy - then this requirement can be acquired using the [data provider](strategy-customization.md#possible-options-for-dataprovider) from within the strategy.

### Advanced Freqtrade Exchange configuration

Advanced options can be configured using the `_ft_has_params` setting, which will override Defaults and exchange-specific behavior.

Available options are listed in the exchange-class as `_ft_has_default`.

For example, to test the order type `FOK` with Kraken, and modify candle limit to 200 (so you only get 200 candles per API call):

```json
"exchange": {
    "name": "kraken",
    "_ft_has_params": {
        "order_time_in_force": ["GTC", "FOK"],
        "ohlcv_candle_limit": 200
        }
    //...
}
```

!!! Warning
    Please make sure to fully understand the impacts of these settings before modifying them.
