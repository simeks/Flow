#include <Core/Common.h>
#include <Core/Platform/Guid.h>
#include <Tools/Testing/Framework.h>

using namespace testing;

// {A32FA86D-D8C5-4696-B84A-DF2A419E26C1}
static const GUID test_guid =
{ 0xa32fa86d, 0xd8c5, 0x4696,{ 0xb8, 0x4a, 0xdf, 0x2a, 0x41, 0x9e, 0x26, 0xc1 } };

TEST_CASE(Guid_to_string)
{
    Guid id;
    memcpy(&id, &test_guid, sizeof(test_guid));

    std::string str = guid::to_string(id);
    ASSERT_EXPR(guid::to_string(id) == "A32FA86D-D8C5-4696-B84A-DF2A419E26C1");
}

TEST_CASE(Guid_from_string)
{
    Guid id_expected;
    memcpy(&id_expected, &test_guid, sizeof(test_guid));

    Guid id = guid::from_string("A32FA86D-D8C5-4696-B84A-DF2A419E26C1");
    ASSERT_EXPR(id == id_expected);
}
