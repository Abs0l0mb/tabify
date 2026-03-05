'use strict';

import {
    Block,
    Menu,
    MenuItem,
    MenuItemData,
    ClientLocation,
    TeamSubmenu
} from '@src/classes';

export class TeamMenuItem extends MenuItem {

    constructor(data: MenuItemData, menu: Menu, private subMenu: TeamSubmenu, parent: Block) {

        super(data, menu, parent);
    }

    /*
    **
    **
    */
    protected route() : void {

        ClientLocation.get().router.route(this.data.path + '?teamId=' + this.subMenu.getSelectedTeamId());
    }
}