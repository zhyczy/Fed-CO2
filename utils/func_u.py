def others_test(version, model, p_model, extra_modules, data_loader, loss_fun, global_prototype, device, flog):
    return


def fedtp_test(idx, model, hnet, data_loader, loss_fun, device):
    hnet.eval()
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    node_weights = hnet(torch.tensor([idx], dtype=torch.long).to(device), True)
    model.load_state_dict(node_weights, strict=False)
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()


def peer_test1(model, p_model, data_loader, loss_fun, device):
    model.eval()
    p_model.eval()
    loss_all, loss_ga, loss_pa, loss_gaad, loss_paad = 0, 0, 0, 0, 0
    total = 0
    correct, correct_g, correct_p, correct_gad, correct_pad = 0, 0, 0, 0, 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        feature_g = model.produce_feature(data)
        feature_p = p_model.produce_feature(data)
        output_g = model.classifier(feature_g)
        output_p = p_model.classifier(feature_p)

        feature_ga = p_model.f_adaptor(feature_g)
        output_ga = p_model.classifier(feature_ga)
        output_pa = p_model.c_adaptor(output_p)

        output = output_g.detach()+output_p
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss_g = loss_fun(output_g, target)
        loss_p = loss_fun(output_p, target)
        loss_ag = loss_fun(output_ga, target)
        loss_ap = loss_fun(output_pa, target)
        loss_ga += loss_g
        loss_pa += loss_p
        loss_gaad += loss_ag
        loss_paad += loss_ap

        pred_g = output_g.data.max(1)[1]
        correct_g += pred_g.eq(target.view(-1)).sum().item()
        pred_p = output_p.data.max(1)[1]
        correct_p += pred_p.eq(target.view(-1)).sum().item()
        pred_ag = output_ga.data.max(1)[1]
        correct_gad += pred_ag.eq(target.view(-1)).sum().item()
        pred_ap = output_pa.data.max(1)[1]
        correct_pad += pred_ap.eq(target.view(-1)).sum().item()

    test_loss = [loss_all/len(data_loader), loss_ga/len(data_loader), loss_pa/len(data_loader), loss_gaad/len(data_loader), loss_paad/len(data_loader)]
    test_acc = [correct/total, correct_g/total, correct_p/total, correct_gad/total, correct_pad/total]
    return test_loss, test_acc


def peer_test2(model, p_model, data_loader, loss_fun, device):
    model.eval()
    p_model.eval()
    loss_all, loss_ga, loss_pa, loss_gaad, loss_paad = 0, 0, 0, 0, 0
    total = 0
    correct, correct_g, correct_p, correct_gad, correct_pad = 0, 0, 0, 0, 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        output_g = model(data)
        output_p = p_model(data)

        feature_ga = model.produce_adapt_feature(data)
        output_ga = p_model.classifier(feature_ga)
        output_pa = p_model.c_adaptor(output_p)

        output = output_g.detach()+output_p
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss_g = loss_fun(output_g, target)
        loss_p = loss_fun(output_p, target)
        loss_ag = loss_fun(output_ga, target)
        loss_ap = loss_fun(output_pa, target)
        loss_ga += loss_g
        loss_pa += loss_p
        loss_gaad += loss_ag
        loss_paad += loss_ap

        pred_g = output_g.data.max(1)[1]
        correct_g += pred_g.eq(target.view(-1)).sum().item()
        pred_p = output_p.data.max(1)[1]
        correct_p += pred_p.eq(target.view(-1)).sum().item()
        pred_ag = output_ga.data.max(1)[1]
        correct_gad += pred_ag.eq(target.view(-1)).sum().item()
        pred_ap = output_pa.data.max(1)[1]
        correct_pad += pred_ap.eq(target.view(-1)).sum().item()

    test_loss = [loss_all/len(data_loader), loss_ga/len(data_loader), loss_pa/len(data_loader), loss_gaad/len(data_loader), loss_paad/len(data_loader)]
    test_acc = [correct/total, correct_g/total, correct_p/total, correct_gad/total, correct_pad/total]
    return test_loss, test_acc


def peer_shabby_adaptor_validate(model, p_model, extra_modules, data_loader, loss_fun, client_idx, device):
    client_num = len(extra_modules)
    assert client_num != 0
    model.eval()
    p_model.eval()
    loss_all, loss_ga, loss_pa, loss_gaad= 0, 0, 0, 0
    total = 0
    correct, correct_g, correct_p, correct_gad= 0, 0, 0, 0
    adapt_loss_dict = {}
    adapt_acc_dict = {}
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        feature_g = model.produce_feature(data)
        feature_p = p_model.produce_feature(data)
        output_g = model.classifier(feature_g)
        output_p = p_model.classifier(feature_p)

        feature_ga = p_model.f_adaptor(feature_g)
        output_ga = p_model.classifier(feature_ga)

        output = output_g.detach()+output_p.detach()

        for idxx in range(client_num):
            if idxx != client_idx:
                adaptor, classifier_adapt = extra_modules[idxx]
                adaptor.eval()
                classifier_adapt.eval()
                feature_adapt = adaptor(feature_g)
                output_adapt = classifier_adapt(feature_adapt)
                loss = loss_fun(output_adapt, target)
                pred = output_adapt.max(1)[1]
                correct_ada = pred.eq(target.view(-1)).sum().item()

                if idxx in adapt_loss_dict.keys():
                    adapt_loss_dict[idxx] += loss.item()
                    adapt_acc_dict[idxx] += correct_ada
                else:
                    adapt_loss_dict[idxx] = loss.item()
                    adapt_acc_dict[idxx] = correct_ada
                output += output_adapt.detach()

        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss_g = loss_fun(output_g, target)
        loss_p = loss_fun(output_p, target)
        loss_ag = loss_fun(output_ga, target)
        loss_ga += loss_g.item()
        loss_pa += loss_p.item()
        loss_gaad += loss_ag.item()

        pred_g = output_g.data.max(1)[1]
        correct_g += pred_g.eq(target.view(-1)).sum().item()
        pred_p = output_p.data.max(1)[1]
        correct_p += pred_p.eq(target.view(-1)).sum().item()
        pred_ag = output_ga.data.max(1)[1]
        correct_gad += pred_ag.eq(target.view(-1)).sum().item()

    for idxx in range(client_num):
        if idxx != client_idx:
            adapt_loss_dict[idxx] = adapt_loss_dict[idxx]/len(data_loader)
            adapt_acc_dict[idxx] = adapt_acc_dict[idxx]/total

    test_loss = [loss_all/len(data_loader), loss_ga/len(data_loader), loss_pa/len(data_loader), loss_gaad/len(data_loader), adapt_loss_dict]
    test_acc = [correct/total, correct_g/total, correct_p/total, correct_gad/total, adapt_acc_dict]
    return test_loss, test_acc


def peer_shabby_validate(model, p_model, data_loader, loss_fun, device):
    model.eval()
    p_model.eval()
    loss_all, loss_ga, loss_pa, loss_gaad = 0, 0, 0, 0
    total = 0
    correct, correct_g, correct_p, correct_gad = 0, 0, 0, 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        feature_g = model.produce_feature(data)
        feature_p = p_model.produce_feature(data)
        output_g = model.classifier(feature_g)
        output_p = p_model.classifier(feature_p)

        feature_ga = p_model.f_adaptor(feature_g)
        output_ga = p_model.classifier(feature_ga)

        output = output_g.detach()+output_p
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss_g = loss_fun(output_g, target)
        loss_p = loss_fun(output_p, target)
        loss_ag = loss_fun(output_ga, target)
        loss_ga += loss_g
        loss_pa += loss_p
        loss_gaad += loss_ag

        pred_g = output_g.data.max(1)[1]
        correct_g += pred_g.eq(target.view(-1)).sum().item()
        pred_p = output_p.data.max(1)[1]
        correct_p += pred_p.eq(target.view(-1)).sum().item()
        pred_ag = output_ga.data.max(1)[1]
        correct_gad += pred_ag.eq(target.view(-1)).sum().item()

        # break

    test_loss = [loss_all/len(data_loader), loss_ga/len(data_loader), loss_pa/len(data_loader), loss_gaad/len(data_loader)]
    test_acc = [correct/total, correct_g/total, correct_p/total, correct_gad/total]
    return test_loss, test_acc


def peer_shabby_validate1(model, p_model, adaptor, data_loader, loss_fun, device):
    model.eval()
    p_model.eval()
    loss_all, loss_ga, loss_pa, loss_gaad = 0, 0, 0, 0
    total = 0
    correct, correct_g, correct_p, correct_gad = 0, 0, 0, 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        feature_g = model.produce_feature(data)
        feature_p = p_model.produce_feature(data)
        output_g = model.classifier(feature_g)
        output_p = p_model.classifier(feature_p)

        feature_ga = adaptor(feature_g)
        output_ga = p_model.classifier(feature_ga)

        output = output_g.detach()+output_p
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss_g = loss_fun(output_g, target)
        loss_p = loss_fun(output_p, target)
        loss_ag = loss_fun(output_ga, target)
        loss_ga += loss_g
        loss_pa += loss_p
        loss_gaad += loss_ag

        pred_g = output_g.data.max(1)[1]
        correct_g += pred_g.eq(target.view(-1)).sum().item()
        pred_p = output_p.data.max(1)[1]
        correct_p += pred_p.eq(target.view(-1)).sum().item()
        pred_ag = output_ga.data.max(1)[1]
        correct_gad += pred_ag.eq(target.view(-1)).sum().item()

    test_loss = [loss_all/len(data_loader), loss_ga/len(data_loader), loss_pa/len(data_loader), loss_gaad/len(data_loader)]
    test_acc = [correct/total, correct_g/total, correct_p/total, correct_gad/total]
    return test_loss, test_acc


def peer_residual_validate(model, p_model, data_loader, loss_fun, device):
    model.eval()
    p_model.eval()
    loss_all, loss_ga, loss_pa, loss_gaad = 0, 0, 0, 0
    total = 0
    correct, correct_g, correct_p, correct_gad = 0, 0, 0, 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        output_g = model(data)
        output_p = p_model(data)

        feature_ga = model.produce_adapt_feature(data)
        output_ga = p_model.classifier(feature_ga)

        output = output_g.detach()+output_p
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss_g = loss_fun(output_g, target)
        loss_p = loss_fun(output_p, target)
        loss_ag = loss_fun(output_ga, target)
        loss_ga += loss_g
        loss_pa += loss_p
        loss_gaad += loss_ag

        pred_g = output_g.data.max(1)[1]
        correct_g += pred_g.eq(target.view(-1)).sum().item()
        pred_p = output_p.data.max(1)[1]
        correct_p += pred_p.eq(target.view(-1)).sum().item()
        pred_ag = output_ga.data.max(1)[1]
        correct_gad += pred_ag.eq(target.view(-1)).sum().item()

    test_loss = [loss_all/len(data_loader), loss_ga/len(data_loader), loss_pa/len(data_loader), loss_gaad/len(data_loader)]
    test_acc = [correct/total, correct_g/total, correct_p/total, correct_gad/total]
    return test_loss, test_acc

