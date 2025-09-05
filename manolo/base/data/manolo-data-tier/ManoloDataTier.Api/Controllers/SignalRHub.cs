using ManoloDataTier.Api.SignalRFeatures.Item.GetItemData;
using Microsoft.AspNetCore.SignalR;

namespace ManoloDataTier.Api.Controllers;

public class SignalRHub : Hub{

    private readonly GetItemDataSignalR _itemDataHandler;

    public SignalRHub(GetItemDataSignalR itemDataHandler){

        _itemDataHandler = itemDataHandler;
    }

    public async Task RequestItemData(string requestId, int dsn){

        await _itemDataHandler.GetItemDataAsync(Clients, Context.ConnectionId, requestId, dsn);
    }

}