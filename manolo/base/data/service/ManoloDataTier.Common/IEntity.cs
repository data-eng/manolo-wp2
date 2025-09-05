namespace ManoloDataTier.Common;

public interface IEntity{

    public                 string Id                { get; set; }
    public                 long   LastChangeDateTime{ get; set; }
    public static abstract string GenerateId();

}